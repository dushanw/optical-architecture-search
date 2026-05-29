import streamlit as st
import jax
import jax.numpy as jnp
import optax
import chromatix.functional as cx
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch

st.set_page_config(page_title="Memetic OAS GUI", layout="wide")

# Our library of discrete optical elements
OPS = [
    "Propagate", 
    "ThinLens", 
    "Identity", 
    "CircularPupil", 
    "SquarePupil", 
    "GaussianPupil",
    "SuperGaussianPupil",
    "RectangularPupil",
    "TukeyPupil",
    "Axicon",
    "SawtoothGrating",
    "SinusoidGrating"
]

@st.cache_data
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

def generate_target_image(shape, dx, spectrum, f1, f2, obj, task):
    if task == "5.1. Amplitude Imaging":
        field = cx.plane_wave(shape=shape, dx=dx, spectrum=spectrum, power=1.0)
        field = cx.amplitude_change(field, obj)
        field = cx.ff_lens(field, f=f1, n=1.0)
        field = cx.ff_lens(field, f=f2, n=1.0)
        return field.intensity
    else:
        # 5.2. Phase Imaging
        # The object is a phase mask. We want the intensity to linearly correspond to the phase.
        # So our target is simply the input mask (scaled appropriately).
        return obj

def simulate_discrete_architecture(z, f, w, arch_indices, shape, dx, spectrum, obj, pad_width, task):
    field = cx.plane_wave(shape=shape, dx=dx, spectrum=spectrum, power=1.0)
    if task == "5.1. Amplitude Imaging":
        field = cx.amplitude_change(field, obj)
    else:
        # 5.2 Phase Imaging: obj is the phase modulation (0 to 1 -> 0 to 2pi)
        field = cx.phase_change(field, obj * 2 * jnp.pi)
    
    z = jnp.abs(z) + 1e-3
    f = jnp.abs(f) + 1e-3
    w = jnp.abs(w) + 1e-3
    
    for i in range(arch_indices.shape[0]):
        def do_prop(fld):
            return cx.transfer_propagate(fld, z=z[i], n=1.0, pad_width=pad_width, mode="same")
        def do_lens(fld):
            return cx.thin_lens(fld, f=f[i], n=1.0)
        def do_id(fld):
            return fld
        def do_circ(fld):
            return cx.circular_pupil(fld, w=w[i])
        def do_sq(fld):
            return cx.square_pupil(fld, w=w[i])
        def do_gauss(fld):
            return cx.gaussian_pupil(fld, w=w[i])
        def do_supergauss(fld):
            return cx.super_gaussian_pupil(fld, w=w[i], n=16.0)
        def do_rect(fld):
            # Let's make it a rectangle with h = w / 2 for some variety
            return cx.rectangular_pupil(fld, h=w[i]/2.0, w=w[i])
        def do_tukey(fld):
            return cx.tukey_pupil(fld, w=w[i])
        def do_axicon(fld):
            return cx.axicon(fld, n_axicon=1.5, slope_angle=w[i]/1000.0)
        def do_grating(fld):
            return cx.sawtooth_grating(fld, n_grating=1.5, period=w[i]/2.0, thickness=w[i]/5.0)
        def do_sin_grating(fld):
            return cx.sinusoid_grating(fld, n_grating=1.5, period=w[i]/2.0, thickness=w[i]/5.0)
        
        field = jax.lax.switch(arch_indices[i], [
            do_prop, do_lens, do_id, 
            do_circ, do_sq, do_gauss, do_supergauss, do_rect, do_tukey,
            do_axicon, do_grating, do_sin_grating
        ], field)
        
    return field.intensity

def loss_fn(z, f, w, arch_indices, shape, dx, spectrum, obj, target_I, pad_width, task):
    I_out = simulate_discrete_architecture(z, f, w, arch_indices, shape, dx, spectrum, obj, pad_width, task)
    if task == "5.1. Amplitude Imaging":
        mse_loss = jnp.mean((I_out - target_I) ** 2) / (jnp.mean(target_I ** 2) + 1e-8)
        return mse_loss
    else:
        # For Phase Imaging, absolute intensity scale might vary significantly.
        # We use a structurally normalized MSE to ensure it's scale-invariant.
        I_out_norm = (I_out - jnp.mean(I_out)) / (jnp.std(I_out) + 1e-8)
        target_I_norm = (target_I - jnp.mean(target_I)) / (jnp.std(target_I) + 1e-8)
        mse_loss = jnp.mean((I_out_norm - target_I_norm) ** 2)
        return mse_loss

def draw_schematic(ax, architecture, title="Optical System"):
    current_z = 0.0
    ax.axhline(0, color='black', linestyle='-.', linewidth=1)
    ax.axvline(0, color='green', linewidth=3, label='Object Plane')
    ax.text(0, 1.2, 'Object', rotation=90, va='bottom', ha='center', color='green')
    
    for op, val in architecture:
        if op == "Propagate":
            z_val = val
            ax.annotate('', xy=(current_z + z_val, 0.5), xytext=(current_z, 0.5),
                        arrowprops=dict(arrowstyle='<|-|>', color='gray'))
            ax.text(current_z + z_val/2, 0.6, f"z={val:.0f}", ha='center')
            current_z += z_val
        elif op == "ThinLens":
            ax.axvline(current_z, ymin=0.1, ymax=0.9, color='blue', linewidth=2)
            ax.annotate('', xy=(current_z, 1.0), xytext=(current_z, 0.8), arrowprops=dict(arrowstyle='->', color='blue'))
            ax.annotate('', xy=(current_z, -1.0), xytext=(current_z, -0.8), arrowprops=dict(arrowstyle='->', color='blue'))
            ax.text(current_z, 1.2, f"f={val:.0f}", rotation=90, va='bottom', ha='center', color='blue')
        elif op in ["CircularPupil", "SquarePupil", "GaussianPupil", "SuperGaussianPupil", "RectangularPupil"]:
            ax.axvline(current_z, ymin=0.3, ymax=0.7, color='orange', linewidth=4)
            ax.text(current_z, 0.8, f"{op[:4]}\nw={val:.0f}", rotation=90, va='bottom', ha='center', color='orange')
        elif op in ["Axicon", "SawtoothGrating", "SinusoidGrating"]:
            ax.axvline(current_z, ymin=0.1, ymax=0.9, color='purple', linewidth=3)
            ax.text(current_z, -1.2, f"{op[:4]}\np={val:.0f}", rotation=90, va='top', ha='center', color='purple')
            
    ax.axvline(current_z, color='red', linewidth=3, label='Image Plane')
    ax.text(current_z, 1.2, 'Sensor', rotation=90, va='bottom', ha='center', color='red')
    
    max_z = current_z if current_z > 0 else 100.0
    ax.set_xlim(-0.1 * max_z, max_z * 1.1)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title(title)

def get_arch_from_params(arch_indices, z, f, w):
    proposed_arch = [OPS[idx] for idx in arch_indices]
    arch = []
    for op, z_val, f_val, w_val in zip(proposed_arch, z, f, w):
        if op == "Propagate":
            arch.append((op, float(z_val)))
        elif op == "ThinLens":
            arch.append((op, float(f_val)))
        elif op in ["CircularPupil", "SquarePupil", "GaussianPupil", "SuperGaussianPupil", "RectangularPupil", "TukeyPupil", "Axicon", "SawtoothGrating", "SinusoidGrating"]:
            arch.append((op, float(w_val)))
        else:
            arch.append((op, 0.0))
    return arch

st.title("Memetic Optical Architecture Search (OAS)")
st.markdown("Evolutionary Algorithm for discrete choices + Gradient Descent for continuous physics parameters.")

task_choice = st.selectbox("Select Imaging Task", 
                           ["5.1. Amplitude Imaging", "5.2. Phase Imaging"], 
                           index=0, 
                           help="Select the goal of the optical architecture search.")

col1, col2 = st.columns([1, 4])

with col1:
    st.header("Search Settings")
    target_f1 = st.number_input("Target f1 (um)", value=4000.0, step=100.0)
    target_f2 = st.number_input("Target f2 (um)", value=4000.0, step=100.0)
    
    num_blocks = st.number_input("Num Elements in Super-Net", value=7, step=1)
    pop_size = st.number_input("Population Size", value=15, step=1)
    gd_steps_per_gen = st.number_input("GD Steps per Generation", value=5, step=1)
    generations = st.slider("Generations", min_value=1, max_value=200, value=60, step=1)
    
    run_btn = st.button("Run Architecture Search", type="primary")

with col2:
    if run_btn:
        shape = (64, 64)
        dx = 5.0
        spectrum = 0.532
        pad_width = 64
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        status_text.text("Loading CIFAR-10...")
        cifar_batch = load_cifar_batch(batch_size=16, img_size=shape[0])
        
        # Init population
        pop_arch = np.random.randint(0, len(OPS), size=(pop_size, num_blocks))
        pop_z = np.random.uniform(1000.0, 8000.0, size=(pop_size, num_blocks))
        pop_f = np.random.uniform(2000.0, 8000.0, size=(pop_size, num_blocks))
        pop_w = np.random.uniform(50.0, 300.0, size=(pop_size, num_blocks))
        
        pop_arch_jnp = jnp.array(pop_arch)
        pop_z_jnp = jnp.array(pop_z)
        pop_f_jnp = jnp.array(pop_f)
        pop_w_jnp = jnp.array(pop_w)
        
        initial_arch = get_arch_from_params(pop_arch_jnp[0], pop_z_jnp[0], pop_f_jnp[0], pop_w_jnp[0])
        
        optimizer = optax.adam(50.0)
        opt_state = jax.vmap(optimizer.init)((pop_z_jnp, pop_f_jnp, pop_w_jnp))
        
        @jax.jit
        def train_indiv(z, f, w, opt_state, arch_indices, a_mask, target_I):
            def l_fn(z_val, f_val, w_val):
                return loss_fn(z_val, f_val, w_val, arch_indices, shape, dx, spectrum, a_mask, target_I, pad_width, task_choice)
            loss, grads = jax.value_and_grad(l_fn, argnums=(0, 1, 2))(z, f, w)
            updates, opt_state = optimizer.update(grads, opt_state, (z, f, w))
            z_new, f_new, w_new = optax.apply_updates((z, f, w), updates)
            return z_new, f_new, w_new, opt_state, loss

        batch_train = jax.jit(jax.vmap(train_indiv, in_axes=(0, 0, 0, 0, 0, None, None)))
        
        status_text.text("Starting Memetic Architecture Search...")
        
        key = jax.random.PRNGKey(42)
        best_loss_history = []
        
        for gen in range(generations):
            key, subkey = jax.random.split(key)
            img_idx = jax.random.randint(subkey, (), 0, cifar_batch.shape[0])
            a_mask = cifar_batch[img_idx]
            
            target_I = generate_target_image(shape, dx, spectrum, target_f1, target_f2, a_mask, task_choice)
            
            # Gradient Descent Phase
            for _ in range(gd_steps_per_gen):
                pop_z_jnp, pop_f_jnp, pop_w_jnp, opt_state, losses = batch_train(pop_z_jnp, pop_f_jnp, pop_w_jnp, opt_state, pop_arch_jnp, a_mask, target_I)
            
            # Evolution Phase
            fitness = np.array(losses)
            best_loss_history.append(float(np.min(fitness)))
            sorted_indices = np.argsort(fitness)
            
            num_elites = max(1, pop_size // 4)
            elite_indices = sorted_indices[:num_elites]
            
            new_arch = [pop_arch_jnp[i] for i in elite_indices]
            new_z = [pop_z_jnp[i] for i in elite_indices]
            new_f = [pop_f_jnp[i] for i in elite_indices]
            new_w = [pop_w_jnp[i] for i in elite_indices]
            new_opt = [jax.tree_util.tree_map(lambda x: x[i], opt_state) for i in elite_indices]
            
            for _ in range(pop_size - num_elites):
                parent_idx = np.random.choice(elite_indices)
                child_arch = np.copy(np.array(pop_arch_jnp[parent_idx]))
                
                # Mutate architecture (20% chance)
                if np.random.rand() < 0.2:
                    mut_idx = np.random.randint(num_blocks)
                    child_arch[mut_idx] = np.random.randint(len(OPS))
                    
                new_arch.append(jnp.array(child_arch))
                new_z.append(pop_z_jnp[parent_idx] + np.random.randn(num_blocks)*100.0)
                new_f.append(pop_f_jnp[parent_idx] + np.random.randn(num_blocks)*100.0)
                new_w.append(pop_w_jnp[parent_idx] + np.random.randn(num_blocks)*10.0)
                new_opt.append(jax.tree_util.tree_map(lambda x: x[parent_idx], opt_state))
                
            pop_arch_jnp = jnp.stack(new_arch)
            pop_z_jnp = jnp.stack(new_z)
            pop_f_jnp = jnp.stack(new_f)
            pop_w_jnp = jnp.stack(new_w)
            opt_state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *new_opt)
            
            progress_bar.progress((gen + 1) / generations)
            status_text.text(f"Generation {gen+1}/{generations} | Best Loss: {fitness[sorted_indices[0]]:.4f}")
        
        st.success("Search Complete!")
        
        # Best individual
        best_idx = sorted_indices[0]
        final_arch = get_arch_from_params(pop_arch_jnp[best_idx], pop_z_jnp[best_idx], pop_f_jnp[best_idx], pop_w_jnp[best_idx])
        
        ideal_arch = [
            ("Propagate", target_f1),
            ("ThinLens", target_f1),
            ("Propagate", target_f1 + target_f2),
            ("ThinLens", target_f2),
            ("Propagate", target_f2)
        ]
        
        st.subheader("Learned Architecture Sequence")
        for i, (op, val) in enumerate(final_arch):
            if op == "Identity":
                st.markdown(f"**Block {i}:** `{op}`")
            else:
                # Based on the operation, determine the parameter unit
                param_name = "z" if op == "Propagate" else ("f" if op == "ThinLens" else "w")
                st.markdown(f"**Block {i}:** `{op}` ({param_name} = {val:.1f} um)")
        
        st.subheader("Architectures Schematics")
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        draw_schematic(axs[0], ideal_arch, "Target Ideal 4f Architecture")
        draw_schematic(axs[1], initial_arch, "Example Initial Super-Net State (Random)")
        draw_schematic(axs[2], final_arch, "Final Learned Architecture (Best Individual)")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Test pass
        test_key, _ = jax.random.split(key)
        test_batch = load_cifar_batch(batch_size=1, img_size=shape[0])
        test_mask = test_batch[0]
        
        target_I = generate_target_image(shape, dx, spectrum, target_f1, target_f2, test_mask, task_choice)
        final_I = simulate_discrete_architecture(pop_z_jnp[best_idx], pop_f_jnp[best_idx], pop_w_jnp[best_idx], pop_arch_jnp[best_idx], shape, dx, spectrum, test_mask, pad_width, task_choice)
        
        st.subheader("Final Output Performance (CIFAR-10 Test Sample)")
        fig_img, axs_img = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axs_img[0].imshow(test_mask, cmap='gray')
        axs_img[0].set_title("Input Image" if task_choice == "5.1. Amplitude Imaging" else "Input Phase Mask")
        fig_img.colorbar(im0, ax=axs_img[0])
        
        im1 = axs_img[1].imshow(target_I.squeeze(), cmap='gray')
        axs_img[1].set_title("Target Ideal Output")
        fig_img.colorbar(im1, ax=axs_img[1])
        
        im2 = axs_img[2].imshow(final_I.squeeze(), cmap='gray')
        axs_img[2].set_title("Learned Architecture Output")
        fig_img.colorbar(im2, ax=axs_img[2])
        
        st.pyplot(fig_img)
    else:
        st.info("Set your parameters on the left and hit run!")
