# Optical Architecture Search

Discrete optical architecture search (OAS) for coherent imaging tasks using [Chromatix](https://github.com/chromatix-team/chromatix) and JAX.

## Features

- **Task 5.1**: Coherent imaging of a thin amplitude object (learn a 4f-like system)
- **Task 5.2**: Phase imaging of a thin object
- **Memetic optimizer**: Genetic algorithm over discrete element choices + gradient descent on continuous parameters (`z`, `f`, `w`)
- **Element library**: Propagate, ThinLens, pupils, axicon, gratings, and more
- **Streamlit GUI**: Interactive search with schematics and CIFAR-10 training masks

## Setup

```bash
python3.12 -m venv venv312
source venv312/bin/activate
cd chromatix && pip install -e . && cd ..
pip install streamlit optax torchvision matplotlib
```

## Run the GUI

```bash
source venv312/bin/activate
streamlit run gui.py
```

## Scripts

| File | Description |
|------|-------------|
| `gui.py` | Main Streamlit app (memetic OAS) |
| `discrete_oas.py` | CLI discrete architecture search |
| `task5_1.py` | Forward 4f simulation (example 5.1) |
| `inverse_design.py` | Early continuous phase-mask inverse design |

## License

MIT — see [LICENSE](LICENSE).

Chromatix is included as a vendored dependency with local patches; upstream Chromatix has its own license.
