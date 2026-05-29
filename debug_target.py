import jax
import jax.numpy as jnp
import chromatix.functional as cx
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch

def load_cifar_batch(batch_size, img_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = jax.tree_util.tree_map(lambda x: jnp.array(x.numpy()), 
                                    next(iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True))))
    return loader[0].squeeze(1)

shape = (64, 64)
dx = 5.0
spectrum = 0.532
f1 = 500.0
f2 = 500.0
pad_width = 64

cifar_batch = load_cifar_batch(1, shape[0])
a_mask = cifar_batch[0]

# Method 1: Discrete physical steps (prop, lens, prop, lens, prop)
field1 = cx.plane_wave(shape=shape, dx=dx, spectrum=spectrum, power=1.0)
field1 = cx.amplitude_change(field1, a_mask)
field1 = cx.transfer_propagate(field1, z=f1, n=1.0, pad_width=pad_width, mode="same")
field1 = cx.thin_lens(field1, f=f1, n=1.0)
field1 = cx.transfer_propagate(field1, z=f1+f2, n=1.0, pad_width=pad_width, mode="same")
field1 = cx.thin_lens(field1, f=f2, n=1.0)
field1 = cx.transfer_propagate(field1, z=f2, n=1.0, pad_width=pad_width, mode="same")
I1 = field1.intensity

# Method 2: Ideal 4f using cx.ff_lens
field2 = cx.plane_wave(shape=shape, dx=dx, spectrum=spectrum, power=1.0)
field2 = cx.amplitude_change(field2, a_mask)
field2 = cx.ff_lens(field2, f=f1, n=1.0)
field2 = cx.ff_lens(field2, f=f2, n=1.0)
I2 = field2.intensity

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Input Mask")
plt.imshow(a_mask, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Method 1: physical steps")
plt.imshow(I1.squeeze(), cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Method 2: ff_lens")
plt.imshow(I2.squeeze(), cmap='gray')
plt.savefig("target_debug.png")
