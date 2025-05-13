"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import glob
import hashlib
import os
import tempfile

import ase
import ase.io
import gradio as gr
import numpy as np
from ase import units
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import LBFGS

from hf_calculator import HFEndpointCalculator


def hash_file(file_path):
    """Generate MD5 hash for a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        data = f.read()
    hasher.update(data)
    return hasher.hexdigest()


EXAMPLE_FILE_HASHES = set(
    [hash_file(file_path) for file_path in glob.glob("examples/*")]
)
MAX_ATOMS = os.environ.get("MAX_ATOMS", 2000)
INFERENCE_ENDPOINT_URL = os.environ["INFERENCE_ENDPOINT_URL"]


def load_check_ase_atoms(structure_file):
    # Validate and write the uploaded file content
    if not structure_file:
        raise gr.Error("You need an input structure file to run a simulation!")

    try:
        atoms = ase.io.read(structure_file)

        if not (all(atoms.pbc) or np.all(~np.array(atoms.pbc))):
            raise gr.Error(
                "Mixed PBC are not supported yet - please set PBC all True or False in your structure before uploading"
            )

        if len(atoms) == 0:
            raise gr.Error("Error: Structure file contains no atoms.")

        if len(atoms) > MAX_ATOMS:
            raise gr.Error(
                f"Error: Structure file contains {len(atoms)}, which is more than {MAX_ATOMS} atoms. Please use a smaller structure for this demo, or run this on a local machine!"
            )

        atoms.positions -= atoms.get_center_of_mass()
        cell_center = atoms.get_cell().sum(axis=0) / 2
        atoms.positions += cell_center

        return atoms
    except Exception as e:
        raise gr.Error(f"Error loading structure with ASE: {str(e)}")


def run_md_simulation(
    structure_file,
    num_steps,
    num_prerelax_steps,
    md_timestep,
    temperature_k,
    md_ensemble,
    task_name,
    total_charge,
    spin_multiplicity,
    explanation: str | None = None,
    oauth_token: gr.OAuthToken | None = None,
    progress=gr.Progress(),
):
    temp_path = None
    traj_path = None
    md_log_path = None
    atoms = None

    if task_name != "OMol":
        total_charge = 0
        spin_multiplicity = 0

    try:

        atoms = load_check_ase_atoms(structure_file)

        # Check if the file is an example
        example = hash_file(structure_file) in EXAMPLE_FILE_HASHES

        atoms.info["charge"] = total_charge
        atoms.info["spin"] = spin_multiplicity

        atoms.calc = HFEndpointCalculator(
            atoms,
            endpoint_url=INFERENCE_ENDPOINT_URL,
            oauth_token=oauth_token,
            example=example,
            task_name=task_name,
        )

        # Attach a progress callback to track in gradio
        interval = 1
        steps = [0]
        expected_steps = num_steps + num_prerelax_steps

        def update_progress():
            steps[-1] += interval
            progress(steps[-1] / expected_steps)

        with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as traj_f:
            traj_path = traj_f.name
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as log_f:
            md_log_path = log_f.name

        # Do a quick pre-relaxation to make sure the system is stable before starting
        opt = LBFGS(atoms, logfile=md_log_path, trajectory=traj_path)
        opt.attach(update_progress, interval=interval)
        opt.run(fmax=0.05, steps=num_prerelax_steps)

        # Initialize the velocity distribution. Since we did a relaxation, half of this
        # will partition to the potential energy right away, so we double the temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k * 2)

        # Initialize the MD integrator
        if md_ensemble == "NVE":
            dyn = VelocityVerlet(atoms, timestep=md_timestep * units.fs)
        elif md_ensemble == "NVT":
            dyn = NoseHooverChainNVT(
                atoms,
                timestep=md_timestep * units.fs,
                temperature_K=temperature_k,
                tdamp=10 * md_timestep * units.fs,
            )
        traj = Trajectory(traj_path, "a", atoms)
        dyn.attach(traj.write, interval=1)
        dyn.attach(update_progress, interval=interval)
        dyn.attach(
            MDLogger(
                dyn,
                atoms,
                md_log_path,
                header=True,
                stress=False,
                peratom=True,
                mode="a",
            ),
            interval=10,
        )

        # Run the simulation!
        dyn.run(num_steps)

        reproduction_script = f"""
import ase.io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import LBFGS
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase import units
from fairchem.core.common.calculator import FAIRChemCalculator

# Read the atoms object from ASE read-able file
atoms = ase.io.read('input_file.traj')

# Set the total charge and spin multiplicity if using the OMol task
atoms.info["charge"] = {total_charge}
atoms.info["spin"] = {spin_multiplicity}

# Set up the calculator
atoms.calc = FAIRChemCalculator(name='UMA-SM-Final', hf_hub_repo_id='facebook/UMA', hf_hub_filename = 'UMA-SM-Final', task_name='{task_name}')

# Do a quick pre-relaxation to make sure the system is stable
opt = LBFGS(atoms, trajectory="relaxation_output.traj")
opt.run(fmax=0.05, steps={num_prerelax_steps})

# Initialize the velocity distribution; we set twice the temperature since we did a relaxation and
# much of the kinetic energy will partition to the potential energy right away
MaxwellBoltzmannDistribution(atoms, temperature_K={temperature_k}*2)

# Initialize the integrator; NVE is shown here as an example, see https://wiki.fysik.dtu.dk/ase/ase/md.html for all options
dyn = VelocityVerlet(atoms, timestep={md_timestep} * units.fs)

# Set up trajectory and MD logger
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=False, peratom=True, mode="w"), interval=10)
traj = Trajectory("md_output.traj"', "w", atoms)
dyn.attach(traj.write, interval=1)

# Run the simulation!
dyn.run({num_steps})
        """

        with open(md_log_path, "r") as md_log_file:
            md_log = md_log_file.read()

        if explanation is None:
            explanation = f"MD simulation of {len(atoms)} atoms for {num_steps} steps with a timestep of {md_timestep} fs at {temperature_k} K in the {md_ensemble} ensemble using the {task_name} UMA task. You submitted this simulation, so I hope you know what's you're looking for or what it means!"

        return traj_path, md_log, reproduction_script, explanation
    except Exception as e:
        raise gr.Error(
            f"Error running MD simulation: {str(e)}. Please try again or report this error."
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        if md_log_path and os.path.exists(md_log_path):
            os.remove(md_log_path)

        if atoms is not None and getattr(atoms, "calc", None) is not None:
            calc = atoms.calc
            atoms.calc = None
            del calc


def run_relaxation_simulation(
    structure_file,
    num_steps,
    fmax,
    task_name,
    total_charge: float,
    spin_multiplicity: float,
    relax_unit_cell,
    explanation: str | None = None,
    oauth_token: gr.OAuthToken | None = None,
    progress=gr.Progress(),
):
    temp_path = None
    traj_path = None
    opt_log_path = None
    atoms = None

    if task_name != "OMol":
        total_charge = 0
        spin_multiplicity = 0

    try:
        atoms = load_check_ase_atoms(structure_file)

        # Check if the file is an example
        example = hash_file(structure_file) in EXAMPLE_FILE_HASHES

        # Center things for consistency in visualization
        atoms.positions -= atoms.get_center_of_mass()
        cell_center = atoms.get_cell().sum(axis=0) / 2
        atoms.positions += cell_center

        atoms.info["charge"] = total_charge
        atoms.info["spin"] = spin_multiplicity

        atoms.calc = HFEndpointCalculator(
            atoms,
            endpoint_url=INFERENCE_ENDPOINT_URL,
            oauth_token=oauth_token,
            example=example,
            task_name=task_name,
        )

        # Set up a trajectory file to keep the results
        with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as traj_f:
            traj_path = traj_f.name
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as log_f:
            opt_log_path = log_f.name

        optimizer = LBFGS(
            FrechetCellFilter(atoms) if relax_unit_cell else atoms,
            trajectory=traj_path,
            logfile=opt_log_path,
        )

        # Attach a progress callback to track in gradio
        interval = 1
        steps = [0]

        def update_progress(steps):
            steps[-1] += interval
            progress(steps[-1] / num_steps)

        optimizer.attach(update_progress, interval=interval, steps=steps)

        optimizer.run(fmax=fmax, steps=num_steps)

        reproduction_script = f"""
import ase.io
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from fairchem.core.common.calcaulator import FAIRChemCalculator

# Read the atoms object from ASE read-able file
atoms = ase.io.read('input_file.traj')

# Set the total charge and spin multiplicity if using the OMol task
atoms.info["charge"] = {total_charge}
atoms.info["spin"] = {spin_multiplicity}

# Set up the calculator
atoms.calc = FAIRChemCalculator(name='UMA-SM-Final', hf_hub_repo_id='facebook/UMA', hf_hub_filename = 'UMA-SM-Final', task_name='{task_name}')

# Initialize the optimizer from ASE
relax_unit_cell = {relax_unit_cell}
optimizer = LBFGS(FrechetCellFilter(atoms) if relax_unit_cell else atoms, trajectory="relaxation_output.traj")

# Run the simulation!
dyn.run({num_steps}, fmax={fmax})
        """

        with open(opt_log_path, "r") as opt_log_file:
            opt_log = opt_log_file.read()

        if explanation is None:
            explanation = f"Relaxation of {len(atoms)} atoms for {num_steps} steps with a force tolerance of {fmax} eV/Å using the {task_name} UMA task. You submitted this simulation, so I hope you know what's you're looking for or what it means!"
        return traj_path, opt_log, reproduction_script, explanation
    except Exception as e:
        raise gr.Error(
            f"Error running relaxation: {str(e)}. Please try again or report this error."
        )
    # Make sure we clean up the temp traj files
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        # if traj_path and os.path.exists(traj_path):
        #     os.remove(traj_path)
        if opt_log_path and os.path.exists(opt_log_path):
            os.remove(opt_log_path)

        if atoms is not None and getattr(atoms, "calc", None) is not None:
            calc = atoms.calc
            atoms.calc = None
            del calc
