# UMA UI - Universal Model for Atoms Interactive Demo

## What This Project Is

A Gradio web application for running interactive molecular dynamics (MD) and structure relaxation simulations using Meta's **Universal Model for Atoms (UMA)** - a large mixture-of-linear-experts graph neural network trained on billions of atoms across five open-science simulation datasets from FAIR Chemistry.

Originally a Hugging Face Space demo (https://facebook-fairchem-uma-demo.hf.space/), this fork has been adapted for **self-hosted deployment on a GPU cluster** (currently a single node with 4x A100 GPUs), with plans to extend to larger HPC clusters (e.g., TACC VISTA) using interactive sessions.

## Architecture

```
app.py                          Main Gradio application (entry point)
simulation_scripts.py           MD and relaxation simulation logic (uses fairchem + ASE)
hf_calculator.py                DEPRECATED - old HF Inference Endpoint calculator (kept for reference)
examples/                       15 example structure files (.cif, .pdb, .xyz, .traj)
figures/                        UMA overview diagram
gradio_molecule3d/              Forked custom Gradio component for 3D molecular visualization
  backend/gradio_molecule3d/    Python component (molecule3d.py) - ASE format conversion
  frontend/                     Svelte + TypeScript + 3dmol.js viewer
    shared/MolecularViewer.svelte   Main 3D viewer (atom selection/dragging WIP)
```

### Key Architectural Change (from upstream)

The upstream demo used remote HF Inference Endpoints via HTTP (`hf_calculator.py`). This fork loads the UMA model **directly on local GPUs** using `fairchem.core`:

```python
# simulation_scripts.py
PREDICTOR = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
atoms.calc = FAIRChemCalculator(PREDICTOR, task_name=task_name.lower())
```

This eliminates the need for OAuth tokens, remote API calls, and HF Inference Endpoints. The old `HFEndpointCalculator` and OAuth validation are commented out / unused.

## Tech Stack

- **Backend:** Python, Gradio 5.x (`<5.30.0`), ASE (Atomic Simulation Environment)
- **ML Model:** `fairchem.core` - `pretrained_mlip` for UMA model loading, `FAIRChemCalculator` as ASE calculator
- **Frontend:** Svelte, TypeScript, 3dmol.js (inside `gradio_molecule3d/` custom component)
- **GPU:** CUDA required - model loaded with `device="cuda"` (grabs gpu:0 by default)

## How to Run

```bash
python app.py
```

On startup this will:
1. Build and install the `gradio_molecule3d` custom component (runs `gradio cc install`, `gradio cc build`, then pip installs the wheel)
2. Load the UMA model onto GPU
3. Launch Gradio on `0.0.0.0:7860` with SSR disabled

For remote/HPC access, SSH port-forward port 7860 to your local machine:
```bash
ssh -L 7860:localhost:7860 user@cluster-node
```

## Key Files and What They Do

### `app.py` (~1105 lines)
The Gradio Blocks UI with three tabs:
1. **UMA Intro** - Model overview + two starter examples
2. **Explore UMA's Capabilities** - ~10 curated examples across chemistry domains
3. **Try UMA on Your Structures** - Custom upload interface

Important additions for cluster use:
- `server_file_path` textbox + `load_server_file_btn` - loads structure files directly from the HPC filesystem without browser upload
- `allowed_paths=[os.path.expanduser("~")]` - allows Gradio to serve files from the user's home directory
- `server_name="0.0.0.0"` - binds to all interfaces for port forwarding access

### `simulation_scripts.py` (~398 lines)
Core simulation engine:
- `run_md_simulation()` - Pre-relaxation (LBFGS), Maxwell-Boltzmann velocity init, NVE/NVT MD
- `run_relaxation_simulation()` - LBFGS optimization with optional unit cell relaxation (FrechetCellFilter)
- `load_check_ase_atoms()` - Structure validation (atom count, PBC consistency, centering)
- `validate_ase_atoms_and_login()` - Input validation (OAuth checks are commented out)

Both simulation functions return: trajectory file, log, ASE reproduction script, explanation text.

### `gradio_molecule3d/` (forked custom component)
Fork of `gradio_molecule3d` with additions:
- PBC (periodic boundary conditions) visualization
- ASE format conversion (3dmol.js doesn't support all formats natively)
- Multi-frame trajectory support
- **WIP:** Atom selection and dragging in `MolecularViewer.svelte` (edit mode toggle, mouse handlers for atom manipulation) - this feature exists but has known bugs (atoms/bonds disappearing)

### `hf_calculator.py` (deprecated)
The old `HFEndpointCalculator` that made HTTP POST requests to HF Inference Endpoints with exponential backoff retry. Not imported anywhere currently.

## UMA Task Types

Each task corresponds to a different chemistry domain and DFT training data:

| Task | Domain | DFT Level | Notes |
|------|--------|-----------|-------|
| **OMol** | Molecules, biomolecules | wB97M-V/def2-TZVPD (ORCA) | Supports charge + spin multiplicity |
| **OMC** | Organic molecular crystals | PBE+D3 (VASP) | charge=0, spin=0 only |
| **OMat** | Inorganic materials | PBE/PBE+U (VASP) | charge=0, spin=0 only |
| **OC20** | Catalysts + adsorbates | RPBE (VASP) | charge=0, spin=0 only |
| **ODAC** | MOFs + CO2/H2O | PBE+D3 (VASP) | charge=0, spin=0 only |

## Supported File Formats

`.xyz`, `.extxyz`, `.cif`, `.pdb`, `.traj` (ASE trajectory)

Constraints: max 2000 atoms (configurable via `MAX_ATOMS` env var), PBC must be all-True or all-False (no mixed).

## Planned / WIP Features

- **Interactive atom editing:** Select, drag, and rotate atoms/clusters in the 3D viewer, then run UMA geometry optimization and observe the optimization trajectory. Currently partially implemented in `MolecularViewer.svelte` but has bugs (disappearing atoms/bonds when selected or dragged).
- **Multi-GPU / larger HPC support:** Currently uses a single GPU on a local node. Plans to deploy via interactive sessions on larger clusters (TACC VISTA, etc.) using `salloc`/`srun` + SSH tunneling.

## Dependencies

`requirements.txt` lists: `gradio<5.30.0`, `numpy`, `ase`, `huggingface_hub`, `backoff`

Additionally required (not in requirements.txt): `fairchem` (FAIR Chemistry library with CUDA support)

## External References

- UMA Paper: https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/
- FAIR Chemistry repo: https://github.com/facebookresearch/fairchem
- UMA model weights: https://huggingface.co/facebook/UMA
- Upstream demo: https://facebook-fairchem-uma-demo.hf.space/
