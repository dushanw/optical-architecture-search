"""
Discrete Optical Architecture Search (OAS)
Using Differentiable Architecture Search (DARTS)

Instead of a fixed sequence of propagations and phase masks, 
this algorithm selects the optical elements from a predefined library.
"""

import jax
import jax.numpy as jnp
import optax
import chromatix.functional as cx
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Our library of discrete optical elements
OPS = ["Propagate", "ThinLens", "Identity"]

def load_cifar_batch(batch_size, img_size):
    """
    Loads a batch of CIFAR-10 images, converts them to grayscale,
    and scales them to be used as amplitude masks in [0, 1].
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # We'll just grab a few images from the training set
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    import torch
    loader = jax.tree_util.tree_map(lambda x: jnp.array(x.numpy()), 
                                    next(iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True))))
    
    # Extract the images and drop the channel dimension so they are (B, H, W)
    images = loader[0].squeeze(1) 
    return images

def generate_target_image(shape, dx, spectrum, f1, f2, a_mask, pad_width):
    """
    Simulates the ideal 4f system to generate the target.
    Using optical FFTs (ff_lens) avoids the severe numerical aliasing 
    (box artifacts) caused by sampling the quadratic phase of a thin lens 
    on a discrete grid with finite resolution.
    """
    field = cx.plane_wave(shape=shape, dx=dx, spectrum=spectrum, power=1.0)
    field = cx.amplitude_change(field, a_mask)
    
    # Ideal 4f system using exact Fraunhofer propagation (optical FFT)
    field = cx.ff_lens(field, f=f1, n=1.0)
    field = cx.ff_lens(field, f=f2, n=1.0)
    
    return field.intensity

def gumbel_softmax(logits, key, temperature=1.0, hard=True):
    gumbels = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-20) + 1e-20)
    y_soft = jax.nn.softmax((logits + gumbels) / temperature)
    
    if hard:
        # Straight-through estimator
        index = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(index, logits.shape[-1])
        y = jax.lax.stop_gradient(y_hard - y_soft) + y_soft
        return y
    return y_soft

def simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width, tau, key):
    field = cx.plane_wave(shape=shape, dx=dx, spectrum=spectrum, power=1.0)
    field = cx.amplitude_change(field, a_mask)
    
    logits = params['arch_logits']
    z = jnp.abs(params['z']) + 1e-3
    f = jnp.abs(params['f']) + 1e-3
    
    keys = jax.random.split(key, logits.shape[0])
    
    for i in range(logits.shape[0]):
        # Hard Gumbel-Softmax: picks exactly one element, avoiding coherent interference
        w = gumbel_softmax(logits[i], keys[i], temperature=tau, hard=True)
        
        u_prop = cx.transfer_propagate(field, z=z[i], n=1.0, pad_width=pad_width, mode="same").u
        u_lens = cx.thin_lens(field, f=f[i], n=1.0).u
        u_id = field.u
        
        u_new = w[0] * u_prop + w[1] * u_lens + w[2] * u_id
        field = field.replace(u=u_new)
        
    return field.intensity

def loss_fn(params, shape, dx, spectrum, a_mask, target_I, pad_width, tau, key):
    I_out = simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width, tau, key)
    mse_loss = jnp.mean((I_out - target_I) ** 2) / jnp.mean(target_I ** 2)
    return mse_loss

def optimize():
    shape = (64, 64)
    dx = 5.0  
    spectrum = 0.532  
    pad_width = 64
    
    target_f1 = 4000.0
    target_f2 = 4000.0
    
    num_blocks = 7 # Give it slightly more capacity than the minimum 5 blocks needed
    
    key = jax.random.PRNGKey(42)
    key_z, key_f, key_logits = jax.random.split(key, 3)
    
    # Initialize parameters
    params = {
        # Architecture routing logits (start neutral with slight noise for symmetry breaking)
        'arch_logits': jax.random.normal(key_logits, (num_blocks, 3)) * 0.01,
        # Physical parameters for the elements (starting in a non-aliasing regime)
        'z': jax.random.uniform(key_z, (num_blocks,), minval=1000.0, maxval=8000.0),
        'f': jax.random.uniform(key_f, (num_blocks,), minval=2000.0, maxval=8000.0),
    }
    
    optimizer = optax.multi_transform(
        {'arch': optax.adam(0.5), 'phys': optax.adam(10.0)},
        param_labels={
            'arch_logits': 'arch',
            'z': 'phys', 'f': 'phys'
        }
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state, a_mask, tau, key):
        target_I = generate_target_image(shape, dx, spectrum, target_f1, target_f2, a_mask, pad_width)
        
        loss, grads = jax.value_and_grad(loss_fn)(params, shape, dx, spectrum, a_mask, target_I, pad_width, tau, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    epochs = 1000
    print("Loading CIFAR-10 data...")
    import torch
    
    # We will use a batch of images to train the architecture
    cifar_batch = load_cifar_batch(batch_size=10, img_size=shape[0])
    
    print("Starting Discrete Architecture Search on CIFAR-10...")
    
    for i in range(epochs):
        key, subkey = jax.random.split(key)
        # Randomly select one image from our loaded batch for stochastic gradient descent
        img_idx = jax.random.randint(subkey, (), 0, cifar_batch.shape[0])
        a_mask = cifar_batch[img_idx]
        
        # Temperature annealing
        tau = max(0.1, 1.0 * (0.98 ** i))
        
        params, opt_state, loss = step(params, opt_state, a_mask, tau, subkey)
        
        if (i + 1) % 100 == 0:
            arch_indices = np.argmax(params['arch_logits'], axis=-1)
            proposed_arch = [OPS[idx] for idx in arch_indices]
            
            print(f"\n--- Epoch {i+1}/{epochs} | Loss: {float(loss):.4f} | Temp: {tau:.3f} ---")
            print("Current Architecture Proposal:")
            for b, (op, z_val, f_val) in enumerate(zip(proposed_arch, params['z'], params['f'])):
                param_str = f"z={float(z_val):.1f}" if op == "Propagate" else (f"f={float(f_val):.1f}" if op == "ThinLens" else "None")
                print(f"  Block {b}: {op} ({param_str})")

    print("\n--- Final Evaluation ---")
    arch_indices = np.argmax(params['arch_logits'], axis=-1)
    proposed_arch = [OPS[idx] for idx in arch_indices]
    print("Final Chosen Architecture:")
    for b, (op, z_val, f_val) in enumerate(zip(proposed_arch, params['z'], params['f'])):
        param_str = f"z={float(z_val):.1f}" if op == "Propagate" else (f"f={float(f_val):.1f}" if op == "ThinLens" else "None")
        print(f"  Block {b}: {op} ({param_str})")
        
    # Test pass on a completely unseen image
    test_key, _ = jax.random.split(key)
    test_batch = load_cifar_batch(batch_size=1, img_size=shape[0])
    test_mask = test_batch[0]
    
    target_I = generate_target_image(shape, dx, spectrum, target_f1, target_f2, test_mask, pad_width)
    
    # Test pass (using tau=0.01 for hard selection)
    final_I = simulate_search_architecture(params, shape, dx, spectrum, test_mask, pad_width, 0.01, test_key)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Object Amplitude Mask")
    plt.imshow(test_mask, cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("Target Intensity (Ideal 4f)")
    plt.imshow(target_I.squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("Learned Architecture Output")
    plt.imshow(final_I.squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.savefig("discrete_oas_output.png")
    print("Saved evaluation plot to discrete_oas_output.png")

if __name__ == "__main__":
    optimize()
