import jax
import jax.numpy as jnp
import chromatix.functional as cx

def simulate_4f_system(shape=(1024, 1024), dx=1.0, spectrum=0.532, f1=100.0, f2=200.0, n=1.0, key_seed=42):
    """
    Implements Example 5.1: Coherent imaging of a thin amplitude object
    using a 4f configuration with two lenses of different focal lengths.
    """
    key = jax.random.PRNGKey(key_seed)
    
    # 1. Input: Electric field from the source is a uniform field
    # (Power is optional, by default normalized to 1.0)
    field = cx.plane_wave(
        shape=shape,
        dx=dx,
        spectrum=spectrum,
        power=1.0,
        scalar=True
    )
    
    # 2. Object Volume: Thin object modeled as a random amplitude mask
    # a(r) in [0, 1]
    amplitude_mask = jax.random.uniform(key, shape, minval=0.0, maxval=1.0)
    
    # The light matter interaction is x_out(r) = x_in(r) * a(r)
    # Using Chromatix's amplitude_change to apply the mask
    field = cx.amplitude_change(field, amplitude_mask)
    
    # 3. 4f Configuration - First Lens (Illumination / Fourier optics)
    # The first lens has focal length f1.
    # ff_lens propagates distance f1 to the lens, applies the lens, and propagates f1 after.
    # So it takes the field from the object plane to the Fourier plane.
    field = cx.ff_lens(field, f=f1, n=n)
    
    # 4. 4f Configuration - Second Lens (Imaging optics)
    # The second lens has focal length f2.
    # Applied to the Fourier plane, it brings us to the Image plane, scaling by M = f2/f1.
    field = cx.ff_lens(field, f=f2, n=n)
    
    # 5. Output: Intensity field at the sensor
    # I_out = |x_out|^2
    I_out = field.intensity
    
    return I_out, amplitude_mask

if __name__ == "__main__":
    shape = (512, 512)
    dx = 1.0  # pixel pitch
    spectrum = 0.532  # wavelength in um
    f1 = 100.0
    f2 = 200.0  # Magnification = f2 / f1 = 2x
    n = 1.0
    
    print(f"Simulating 4f system with f1={f1}, f2={f2} (Magnification {f2/f1}x)")
    I_out, a_mask = simulate_4f_system(shape, dx, spectrum, f1, f2, n)
    
    # In a 4f system, the image should be an inverted, magnified version of the object.
    print(f"Output Intensity shape: {I_out.shape}")
    print(f"Max intensity: {float(jnp.max(I_out)):.6e}")
    print(f"Mean intensity: {float(jnp.mean(I_out)):.6e}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Object Amplitude Mask")
    plt.imshow(a_mask, cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Output Intensity (Image Plane)")
    plt.imshow(I_out.squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.savefig("task5_1_output.png")
    print("Saved plot to task5_1_output.png")
    print("Simulation complete!")
