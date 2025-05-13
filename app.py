"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

from simulation_scripts import run_md_simulation, run_relaxation_simulation

DEFAULT_MOLECULAR_REPRESENTATIONS = [
    {
        "model": 0,
        "chain": "",
        "resname": "",
        "style": "sphere",
        "color": "Jmol",
        "around": 0,
        "byres": False,
        "scale": 0.3,
    },
    {
        "model": 0,
        "chain": "",
        "resname": "",
        "style": "stick",
        "color": "Jmol",
        "around": 0,
        "byres": False,
        "scale": 0.2,
    },
]
DEFAULT_MOLECULAR_SETTINGS = {
    "backgroundColor": "white",
    "orthographic": False,
    "disableFog": False,
}


def main():
    input_structure = gr.File(
        label="ASE-compatible structure",
        file_types=[".cif", ".pdb", ".xyz", ".traj", "INCAR", "POSCAR"],
        height=150,
    )
    output_traj = gr.File(
        label="Simulation Trajectory (ASE traj file)",
        interactive=False,
        height=150,
    )
    input_visualization = Molecule3D(
        label="Input Visualization",
        reps=DEFAULT_MOLECULAR_REPRESENTATIONS,
        config=DEFAULT_MOLECULAR_SETTINGS,
        render=False,
        inputs=[input_structure],
        value=lambda x: x,
        interactive=False,
    )
    md_steps = gr.Slider(minimum=1, maximum=500, value=100, label="MD Steps")
    prerelax_steps = gr.Slider(
        minimum=0, maximum=100, value=20, label="Pre-Relaxation Steps"
    )
    temperature_k = gr.Slider(
        minimum=0.0,
        maximum=1500.0,  # Adjusted max value for temperature
        value=300.0,
        label="Temp [K]",
    )
    md_timestep = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, label="Timestep [fs]")
    md_ensemble = gr.Radio(
        label="Thermodynamic Ensemble", choices=["NVE", "NVT"], value="NVE"
    )
    optimization_steps = gr.Slider(
        minimum=1, maximum=500, value=300, step=1, label="Max Steps"
    )
    fmax = gr.Slider(value=0.05, minimum=0.001, maximum=0.5, label="Opt. Fmax [eV/Å]")

    task_name = gr.Radio(
        value="OMol", choices=["OMol", "OMC", "OMat", "OC20", "ODAC"], label="Task Name"
    )

    gr.Markdown("OMol-specific settings for total charge and spin multiplicity")
    total_charge = gr.Slider(
        value=0, label="Total Charge", minimum=-10, maximum=10, step=1
    )
    spin_multiplicity = gr.Slider(
        value=1, maximum=11, minimum=1, step=1, label="Spin Multiplicity "
    )
    relax_unit_cell = gr.Checkbox(value=False, label="Relax Unit Cell")

    md_button = gr.Button("Run MD Simulation")
    optimization_button = gr.Button("Run Optimization")

    output_structure = Molecule3D(
        label="Simulation Visualization",
        reps=DEFAULT_MOLECULAR_REPRESENTATIONS,
        config=DEFAULT_MOLECULAR_SETTINGS,
        render=False,
        elem_classes="structures",
        height=500,
        inputs=[output_traj],
        value=lambda x: x,
        interactive=False,
    )

    explanation = gr.Markdown()
    explanation_buffer = gr.Markdown()

    output_text = gr.Code(lines=20, max_lines=30, label="Log", interactive=False)

    reproduction_script = gr.Code(
        interactive=False,
        max_lines=30,
        language="python",
        label="ASE Reproduction Script",
    )

    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Column(variant="panel"):
                    gr.Markdown("# Meta's Universal Model for Atoms (UMA) Demo")

                    with gr.Tab("1. UMA Intro"):
                        gr.Image(
                            "figures/uma_overview_figure.svg",
                            label="UMA Overview",
                            show_share_button=False,
                            show_download_button=False,
                        )

                        gr.Markdown(
                            "This is UMA! It is a large mixture-of-linear-experts graph network model trained on billions of atoms across five open-science simulation datasets released by the FAIR Chemistry team over the past 5 years. If you give it an input structure and which task you're interested in modeling in, it will output the energy, forces, and stress which you can use for a molecular simulation! Try one of these examples to see what it can do."
                        )
                        with gr.Row():
                            gr.Examples(
                                examples=[
                                    [
                                        str(
                                            Path(__file__).parent
                                            / "./examples/metal_cplx.pdb"
                                        ),
                                        100,
                                        20,
                                        1.0,
                                        300.0,
                                        "NVE",
                                        "OMol",
                                        1,
                                        6,
                                        "Congratulations, you just ran your first UMA simulation! This is a molecular dynamics simulation of a transition metal complex that shows the atoms vibrating from thermal motion. Now try running some of the examples in the next tab to more thoroughly explore how UMA can be applied to different domains and types of structures.",
                                    ],
                                    [
                                        str(
                                            Path(__file__).parent
                                            / "./examples/inorganic_crystal.cif"
                                        ),
                                        200,
                                        20,
                                        1.0,
                                        300.0,
                                        "NVE",
                                        "OMat",
                                        0,
                                        1,
                                        "Congratulations, you just ran your first UMA simulation! This is a molecular dynamics simulation of an inorganic crystal structure that shows the atoms vibrating from thermal motion. Now try running some of the examples in the next tab to more thoroughly explore how UMA can be applied to different domains and types of structures.",
                                    ],
                                ],
                                example_labels=[
                                    "A transition metal complex",
                                    "An inorganic crystal",
                                ],
                                inputs=[
                                    input_structure,
                                    md_steps,
                                    prerelax_steps,
                                    md_timestep,
                                    temperature_k,
                                    md_ensemble,
                                    task_name,
                                    total_charge,
                                    spin_multiplicity,
                                    explanation_buffer,
                                ],
                                outputs=[
                                    output_traj,
                                    output_text,
                                    reproduction_script,
                                    explanation,
                                ],
                                fn=run_md_simulation,
                                run_on_click=True,
                                cache_examples=True,
                                label="Molecular Dynamics Examples",
                            )

                        gr.Markdown(
                            """
                            
                            When you've run your first UMA simulation, click on the next tab above to explore the UMA model in more detail and see how it works across many different domains/examples!
                            """
                        )

                    with gr.Tab("2. Explore UMA's capabilities"):
                        gr.Markdown(
                            """
    These next examples are designed to show how the UMA model can be applied to different domains and types of structures, and how different model inputs can impact the results!
    * As you try each one, look at how the inputs change below, and the simulation outputs change on the right
    * Each UMA task corresponds to a different domain of chemistry, and a different Density Functional Theory (DFT) code and level of theory that was used for the training data.
    * Feel free to try changing some of the settings below and re-run the simulations to see how the results can vary!
    """
                        )
                        gr.Examples(
                            examples=[
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/metal_cplx.pdb"
                                    ),
                                    100,
                                    20,
                                    1.0,
                                    300.0,
                                    "NVE",
                                    "OMol",
                                    1,
                                    6,
                                    "This metal complex showcases the UMA's ability to handle complicated transition metal complexes with ligands, spin, and charge.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/organic_molecular_crystal.cif"
                                    ),
                                    200,
                                    20,
                                    1.0,
                                    300.0,
                                    "NVE",
                                    "OMC",
                                    0,
                                    1,
                                    "This organic crystal showcases the UMA's ability to handle organic molecular crystals using the OMC task, using a random packing of a molecule from OE62. You could also try using the OMol or OMat tasks and see how the simulations differ!",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/inorganic_crystal.cif"
                                    ),
                                    200,
                                    20,
                                    1.0,
                                    300.0,
                                    "NVE",
                                    "OMat",
                                    0,
                                    1,
                                    "This inorganic crystal from the Materials Project showcases the UMA's ability to handle inorganic materials using the OMat task. You should be careful with the output energies since the analysis might require careful application of the Materials Project GGA/GGA+U correction schemes.",
                                ],
                                [
                                    str(Path(__file__).parent / "./examples/HEA.cif"),
                                    200,
                                    20,
                                    1.0,
                                    300.0,
                                    "NVE",
                                    "OMat",
                                    0,
                                    1,
                                    "High-entropy alloys are materials with 4 or more elements, and this example shows the OMat model can also be applied to HEAs. This particular HEA is equimolar Cr/Fe/Ni/Sn, but could exist in other lattice configurations. ",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/catalyst.traj"
                                    ),
                                    200,
                                    20,
                                    1.0,
                                    300.0,
                                    "NVE",
                                    "OC20",
                                    0,
                                    1,
                                    "This example shows the OC20 model can be applied to small molecules on a catalyst surface, and that OC20 has recently been extended with datasets for multi-adsorbate interactions. Notice some of the adsorbates wrap around the periodic boundary conditions. Some of the adsorbates are weakly bound and desorbing from the surface based on this starting configuration.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/gold_nanoparticle_crystal.cif"
                                    ),
                                    200,
                                    20,
                                    1.0,
                                    300.0,
                                    "NVE",
                                    "OMat",
                                    0,
                                    1,
                                    "This example is an experimentally solved crystal structure for a thiolate protected gold nanocluster from an[open-access academic paper](https://pubs.rsc.org/en/content/articlelanding/2016/sc/c5sc02134k) and [available in the COD](https://www.crystallography.net/cod/1540567.html). This is a fun example because it combines small molecules, inorganic materials, and surface chemistry, so it's not so clear which task to use. Try the OMat, OMol, and OC20 tasks to see how the results differ! Further, the experimental paper reported different crystal structures based on the charge and spin multiplicity, so try changing the charge and spin multiplicity to see how the results differ.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/ethylene_carbonate.xyz"
                                    ),
                                    500,
                                    20,
                                    1.0,
                                    1000.0,
                                    "NVE",
                                    "OMol",
                                    0,
                                    1,
                                    "Ethylene carbonate is a common electrolyte in batteries, and an important precursor to the solid/electrolyte interface (SEI) formation. This example is the neutral version of ethylene carbonate and it is quite stable even at high temperatures! Try the radical anion version of ethylene carbonate in the next example.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/ethylene_carbonate.xyz"
                                    ),
                                    500,
                                    20,
                                    1.0,
                                    1000.0,
                                    "NVE",
                                    "OMol",
                                    -1,
                                    2,
                                    "With charge of -1 and spin multiplicity of 2, this radical anion version of ethylene carbonate is much less stable than the neutral version and can undergo spontaneous ring-opening at high temperatures, which initiates the formation of a battery's solid/electrolyte interface (SEI). If the simulation doesn't show a ring opening reaction, try clicking the molecular dynamic buttons again to see another short simulation. Only OMol currently supports arbitrary charge/spin multiplicity.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent / "./examples/protein.pdb"
                                    ),
                                    200,
                                    20,
                                    1.0,
                                    300.0,
                                    "NVE",
                                    "OMol",
                                    0,
                                    1,
                                    "This is a solvated structure of a small protein fragment using the OMol task. Very little is likely happen in a very short MD simulations. Try increasing the number of MD steps to see how the protein moves. You can also try using the OMat task, but be careful with the results since the OMat task is not trained on solvated systems.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/MOF_CO2.traj"
                                    ),
                                    200,
                                    20,
                                    1.0,
                                    300.0,
                                    "NVE",
                                    "ODAC",
                                    0,
                                    1,
                                    "This is a metal organic framework (MOF) structure using the ODac task. You might study structures like if designing MOFs for direct air capture calculations. Look carefully for the red/gray CO2 molecule in the pores of the MOF! You can also try the OMol and OMat tasks to see if the results differ.",
                                ],
                            ],
                            example_labels=[
                                "Simulate a  transition metal complex",
                                "Simulate an organic molecular crystal",
                                "Simulate an inorganic crystal",
                                "Simulate a high-entropy alloy",
                                "Simulate a catalyst surface/adsorbate",
                                "Simulate a ligated gold nanoparticle crystal",
                                "Simulate a neutral ethylene carbonate",
                                "Simulate a radical anion ethylene carbonate",
                                "Simulate a solvated protein",
                                "Simulate CO2 in a metal organic framework",
                            ],
                            inputs=[
                                input_structure,
                                md_steps,
                                prerelax_steps,
                                md_timestep,
                                temperature_k,
                                md_ensemble,
                                task_name,
                                total_charge,
                                spin_multiplicity,
                                explanation_buffer,
                            ],
                            outputs=[
                                output_traj,
                                output_text,
                                reproduction_script,
                                explanation,
                            ],
                            fn=run_md_simulation,
                            run_on_click=True,
                            cache_examples=True,
                            label="Molecular Dynamics Examples",
                        )

                        gr.Examples(
                            examples=[
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/bis(EDA)Cu.xyz"
                                    ),
                                    300,
                                    0.05,
                                    "OMol",
                                    1,
                                    1,
                                    False,
                                    "This is a super fun example of a transition metal complex that changes its geometry based on the assumed charge/spin multiplicity and actually has different local minima. It highlights the ability of the OMol task to handle these additional inputs. In this first example it forms a tetragonal geometry - try clicking on the next example (with charge=+2 and spin multiplicity 2) to see the other local minima!",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/bis(EDA)Cu.xyz"
                                    ),
                                    300,
                                    0.05,
                                    "OMol",
                                    2,
                                    2,
                                    False,
                                    "This is a super fun example of a transition metal complex that changes its geometry based on the assumed charge/spin multiplicity and actually has different local minima. In contrast to the previous example (charge=+1 and spin multiplicity 1), this one forms a square planar geometry.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/metal_cplx.pdb"
                                    ),
                                    300,
                                    0.05,
                                    "OMol",
                                    1,
                                    6,
                                    False,
                                    "This metal complex showcases the UMA's ability to handle complicated transition metal complexes with ligands, spin, and charge.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/organic_molecular_crystal.cif"
                                    ),
                                    300,
                                    0.05,
                                    "OMC",
                                    0,
                                    1,
                                    True,
                                    "This organic crystal showcases the UMA's ability to handle organic molecular crystals using the OMC task, using a random packing of a molecule from OE62. You could also try using the OMol or OMat tasks and see how the simulations differ!",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/inorganic_crystal.cif"
                                    ),
                                    300,
                                    0.05,
                                    "OMat",
                                    0,
                                    1,
                                    True,
                                    "This inorganic crystal from the Materials Project showcases the UMA's ability to handle inorganic materials using the OMat task. You should be careful with the output energies since the analysis might require careful application of the Materials Project GGA/GGA+U correction schemes.",
                                ],
                                [
                                    str(Path(__file__).parent / "./examples/HEA.cif"),
                                    300,
                                    0.05,
                                    "OMat",
                                    0,
                                    1,
                                    True,
                                    "High-entropy alloys are materials with 4 or more elements, and this example shows the OMat model can also be applied to HEAs. This particular HEA is equimolar Cr/Fe/Ni/Sn, but could exist in other lattice configurations. ",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/catalyst.traj"
                                    ),
                                    300,
                                    0.05,
                                    "OC20",
                                    0,
                                    1,
                                    False,
                                    "This example shows the OC20 model can be applied to small molecules on a catalyst surface, and that OC20 has recently been extended with datasets for multi-adsorbate interactions. Notice some of the adsorbates wrap around the periodic boundary conditions. Some of the adsorbates are weakly bound and desorbing from the surface based on this starting configuration.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent / "./examples/protein.pdb"
                                    ),
                                    300,
                                    0.05,
                                    "OMol",
                                    0,
                                    1,
                                    True,
                                    "This is a solvated structure of a small protein fragment using the OMol task. Very little is likely happen in a very short MD simulations. Try increasing the number of MD steps to see how the protein moves. You can also try using the OMat task, but be careful with the results since the OMat task is not trained on solvated systems.",
                                ],
                                [
                                    str(
                                        Path(__file__).parent
                                        / "./examples/MOF_CO2_2H2O.traj"
                                    ),
                                    300,
                                    0.05,
                                    "ODAC",
                                    0,
                                    1,
                                    False,
                                    "This is a metal organic framework (MOF) structure using the ODac task. You might study structures like if designing MOFs for direct air capture (DAC) calculations. Look carefully for the co-adsorption between CO2 and two water molecules in the pore, which you might study if you were interested in the effect of humidity on DAC performance! You can also try the OMol and OMat tasks to see if the results differ.",
                                ],
                            ],
                            example_labels=[
                                "Relax bis(EDA)Cu TM complex with charge=1, spin=1",
                                "Relax bis(EDA)Cu TM complex with charge=2, spin=2",
                                "Relax a  transition metal complex w ligands",
                                "Relax an organic molecular crystal",
                                "Relax an inorganic crystal",
                                "Relax a high-entropy alloy",
                                "Relax a catalyst surface/adsorbate",
                                "Relax a solvated protein",
                                "Relax co-adsorbed H2O/CO2 in a metal organic framework",
                            ],
                            inputs=[
                                input_structure,
                                optimization_steps,
                                fmax,
                                task_name,
                                total_charge,
                                spin_multiplicity,
                                relax_unit_cell,
                                explanation_buffer,
                            ],
                            outputs=[
                                output_traj,
                                output_text,
                                reproduction_script,
                                explanation,
                            ],
                            fn=run_relaxation_simulation,
                            run_on_click=True,
                            cache_examples=True,
                            label="Try an example!",
                        )

                        gr.Markdown(
                            "Once you understand how the UMA model can be applied to different types of molecules and materials, the final tab above will help you try it out with your own structures! "
                        )

                    with gr.Tab("3. Try UMA on your structures!"):
                        gr.Markdown(
                            """
                            As the final step of the demo, try running your own structure through the UMA model!

                            To use a custom input structure with this demo:
                            1. [Request gated model access.](https://huggingface.co/facebook/UMA) Requests for model access are typically processed within a matter of minutes.
                            2. Login to Hugging Face using the "Sign in with Hugging Face button" in the .                       
                            3. Then upload a structure file below and click run!

                            * Note that uploaded structure will be stored by this demo to analyze model usage and identify domains where model accuracy can be improved.
                            * If you get a redirect error when logging in, please try visiting the direct demo url (https://facebook-fairchem-uma-demo.hf.space/) and try again
                            * Your structure should be in a format supported by ASE 3.25, including .xyz, .cif, .pdb, ASE .traj, INCAR, or POSCAR.
                            * Your structure should either have periodic boundary conditions (PBC) all True, or all False. Support for mixed PBC may be added in the future. 
                            """
                        )

                with gr.Sidebar(open=True):
                    gr.Markdown("## Learn more about UMA")
                    with gr.Accordion("What is UMA?", open=False):
                        gr.Markdown(
                            """
    * UMA models predict motion and behavior at the atomic scale, ultimately reducing the development cycle in molecular and materials discovery and unlocking new possibilities for innovation and impact.  
    * UMA models are based on Density Functional Theory (DFT) training datasets. DFT simulations are a commonly used quantum chemistry method to simulate and understand behavior at the atomic scale.
    * UMA models are large mixture-of-linear-experts graph networks models trained on billions of atoms across five open-science simulation datasets released by the FAIR Chemistry team over the past 5 years. This demo uses the small UMA model with 146M total parameters, 32 experts, and 6M active parameters at any time to predict across all of these domains.  

    Read the UMA paper for details or download the UMA model and FAIR Chemistry repository to use this yourself!
    """
                        )
                    with gr.Accordion("Should I trust UMA?", open=False):
                        gr.Markdown(
                            """
    * The UMA model paper contains rigorous accuracy benchmarks on a number of validation sets across chemistry and materials science. As of model release the UMA model was at or near the state-of-the-art for generalization machine learning potentials.  Read the UMA paper for details.
    * Rigorously predicting when AI/ML models will extrapolate (or not) to new domains is an ongoing research area. The best approach is to find or build benchmarks that are similar to the questions you are studying, or be prepared to run some DFT simulations on predictions to validate results on a sample of structures that are relevant to your research problem. 
    """
                        )
                    with gr.Accordion("Why does this matter?", open=False):
                        gr.Markdown(
                            """
    * Many important technological challenges, including developing new molecules to accelerate industrial progress and discovering new materials for energy storage and climate change mitigation, require scientists and engineers to design at the atomic scale.
    * Traditional experimental discovery and design processes are extremely time consuming and often take decades from ideation to scaled manufacturing. 
    * Meta's Fundamental AI Research Lab (FAIR) is drastically accelerating this process by developing accurate and generalizable machine learning models, building on work by academic, industrial, and national lab collaborators. 
    """
                        )
                    with gr.Accordion("Open source packages in this demo", open=False):
                        gr.Markdown(
                            """
    * The model code is available on github at [FAIR chemistry repo](https://github.com/facebookresearch/fairchem)
    * This demo builds on a number of great open source packages like [gradio_molecule3d](https://huggingface.co/spaces/simonduerr/gradio_molecule3d), [3dmol.js](https://3dmol.csb.pitt.edu/), [ASE](https://wiki.fysik.dtu.dk/ase/), and many others!
    """
                        )
                    with gr.Accordion("How fast are these simulations?", open=False):
                        gr.Markdown(
                            """
                            * Each simulation you see would take days or weeks using a traditional quantum chemistry simulation, but UMA can do it in seconds or minutes! 
                            * Examples in the demo are cached ahead of time so they should load right away, but if you run a custom simulation you'll see a progress bar while the simulation runs.'
                            """
                        )

                gr.Markdown("## Simulation inputs")

                with gr.Column(variant="panel"):
                    gr.Markdown("### 1. Example structure (or upload your own!)")
                    with gr.Row():
                        with gr.Column():
                            input_structure.render()

                            gr.LoginButton(size="large")

                            gr.Markdown(
                                """
                            To use your own structures, you need access to the [gated UMA model repository](https://huggingface.co/facebook/UMA) and you need to login with the button above. 
                            * See the final tab above '3. Try UMA with your own structures!' for more details and debugging steps!
                            * Note that uploaded structure will be stored by this demo to analyze model usage and identify domains where model accuracy can be improved.
                            * If you get a redirect error when logging in, please try visiting the direct demo url (https://facebook-fairchem-uma-demo.hf.space/) and try again
                            
                            """
                            )
                        with gr.Column(scale=3):
                            input_visualization.render()

                with gr.Column(variant="panel"):
                    gr.Markdown("### 2. Choose the UMA Model Task")
                    with gr.Row():
                        with gr.Column():
                            task_name.render()

                            with gr.Row():
                                total_charge.render()
                                spin_multiplicity.render()

                        with gr.Column(scale=2):
                            with gr.Tabs() as task_name_tabs:
                                with gr.TabItem("OMol", id=0):
                                    gr.Markdown(
                                        """

                                        OMol25 comprises over 100 million calculations covering small molecules, biomolecules, metal complexes, and electrolytes.

                                        **Relevant applications:** Biology, organic chemistry, protein folding, small-molecule pharmaceuticals, organic liquid properties, homogeneous catalysis

                                        **Level of theory:** wB97M-V/def2-TZVPD as implemented in ORCA6, including many-body dispersion. All solvation should be explicit. 

                                        **Additional inputs:** total charge and spin multiplicity. If you don't know what these are, you should be very careful if modeling charged or open-shell systems. This can be used to study radical chemistry or understand the impact of magnetic states on the structure of a molecule.

                                        **Caveats:** All training data is aperiodic, so any periodic systems should be treated with some caution. Probably won't work well for inorganic materials. 
                                        """
                                    )
                                with gr.TabItem("OMC", id=1):
                                    gr.Markdown(
                                        """

                                        OMC25 comprises ~25 million calculations of organic molecular crystals from random packing of OE62 structures into various 3D unit cells. 

                                        **Relevant applications:** Pharmaceutical packaging, bio-inspired materials, organic electronics, organic LEDs

                                        **Level of theory:** PBE+D3 as implemented in VASP.

                                        **Additional inputs:** UMA has not seen varying charge or spin multiplicity for the OMC task, and expects total_charge=0 and spin multiplicity=0 as model inputs. 

                                        """
                                    )

                                with gr.TabItem("OMat", id=2):
                                    gr.Markdown(
                                        """

                                        OMat24 comprises >100 million calculations or inorganic materials collected from many open databases like Materials Project and Alexandria, and randomly sampled far from equilibria. 

                                        **Relevant applications:** Inorganic materials discovery, solar photovoltaics, advanced alloys, superconductors, electronic materials, optical materials

                                        **Level of theory:** PBE/PBE+U as implemented in VASP using Materials Project suggested settings, except with VASP6 pseudopotentials. No dispersion.

                                        **Additional inputs:** UMA has not seen varying charge or spin multiplicity for the OMat task, and expects total_charge=0 and spin multiplicity=0 as model inputs. 

                                        **Caveats:** Spin polarization effects are included, but you can't select the magnetic state. Further, OMat24 did not fully sample possible spin states in the training data.
                                        """
                                    )

                                with gr.TabItem("OC20", id=3):
                                    gr.Markdown(
                                        """

                                        OC20 comprises >100 million calculations of small molecules adsorbed on catalyst surfaces formed from materials in the Materials Project. 

                                        **Relevant applications:** Renewable energy, catalysis, fuel cells, energy conversion, sustainable fertilizer production, chemical refining, plastics synthesis/upcycling

                                        **Level of theory:** RPBE as implemented in VASP, with VASP5.4 pseudopotentials. No dispersion.

                                        **Additional inputs:** UMA has not seen varying charge or spin multiplicity for the OC20 task, and expects total_charge=0 and spin multiplicity=0 as model inputs.  

                                        **Caveats:** No oxides or explicit solvents are included in OC20. The model works surprisingly well for transition state searches given the nature of the training data, but you should be careful. RPBE works well for small molecules, but dispersion will be important for larger molecules on surfaces. 
                                        """
                                    )

                                with gr.TabItem("ODAC", id=4):
                                    gr.Markdown(
                                        """

                                        ODAC23 comprises >10 million calculations of CO2/H2O molecules adsorbed in Metal Organic Frameworks sampled from various open databases like CoreMOF.

                                        **Relevant applications:** Direct air capture, carbon capture and storage, CO2 conversion, catalysis

                                        **Level of theory:** PBE+D3 as implemented in VASP, with VASP5.4 pseudopotentials. 

                                        **Additional inputs:** UMA has not seen varying charge or spin multiplicity for the ODAC task, and expects total_charge=0 and spin multiplicity=0 as model inputs.  

                                        **Caveats:** The ODAC23 dataset only contains CO2/H2O water absorption, so anything more than might be inaccurate (e.g. hydrocarbons in MOFs). Further, there is a limited number of bare-MOF structures in the training data, so you should be careful if you are using a new MOF structure.
                                        """
                                    )

                with gr.Column(variant="panel"):
                    gr.Markdown("### 3. Run Your Simulation")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Molecular Dynamics")
                            prerelax_steps.render()
                            md_steps.render()
                            temperature_k.render()
                            md_timestep.render()
                            md_ensemble.render()
                            md_button.render()

                        with gr.Column():
                            gr.Markdown("### Relaxation")
                            optimization_steps.render()
                            fmax.render()
                            relax_unit_cell.render()
                            optimization_button.render()

            with gr.Column(variant="panel", elem_id="results", min_width=500):
                gr.Markdown("## UMA Simulation Results")
                with gr.Tab("Visualization"):
                    output_structure.render()
                    with gr.Accordion(
                        "What should I look for in this simulation?", open=True
                    ):
                        explanation.render()
                    with gr.Accordion("Model Disclaimers", open=False):
                        gr.Markdown(
                            """
    * While UMA represents a step forward in terms of having a single model that works across chemistry and materials science, we know the model has limitations and weaknesses and there will be cases where the model fails to produce an accurate simulation.
    * Ab-initio calculations are not perfect. You should always consider the limitations of the level of theory, the code, and the pseudopotentials. 
    """
                        )
                    with gr.Accordion(
                        "How long should I wait for this simulation?", open=False
                    ):
                        gr.Markdown(
                            """
* Every calculation uses a pool of GPUs to process simulations for all current users. You can achieve much higher performance with a dedicated GPU and MD-mode enabled. 
* Most simulation should finish within a few minutes. Example results are cached, and if you are running a custom simulation you can follow the progress bar
* if you don't see progress or the simulation takes more than ~5min, probably there was an error and please try submitting again. 
* If you notice any issues please submit them as issues on the [FAIR Chemistry GitHub](https://github.com/facebookresearch/fairchem).
"""
                        )

                    output_traj.render()
                with gr.Tab("Log"):
                    output_text.render()
                with gr.Tab("Script"):
                    reproduction_script.render()

        md_button.click(
            run_md_simulation,
            inputs=[
                input_structure,
                md_steps,
                prerelax_steps,
                md_timestep,
                temperature_k,
                md_ensemble,
                task_name,
                total_charge,
                spin_multiplicity,
            ],
            outputs=[output_traj, output_text, reproduction_script, explanation],
            scroll_to_output=True,
            concurrency_limit=16,
            concurrency_id="simulation_queue",
        )
        optimization_button.click(
            run_relaxation_simulation,
            inputs=[
                input_structure,
                optimization_steps,
                fmax,
                task_name,
                total_charge,
                spin_multiplicity,
                relax_unit_cell,
            ],
            outputs=[output_traj, output_text, reproduction_script, explanation],
            scroll_to_output=True,
            concurrency_id="simulation_queue",
        )

        # Change the tab based on the current task name
        task_name.input(
            lambda x: gr.Tabs(
                selected={"OMol": 0, "OMC": 1, "OMat": 2, "OC20": 3, "ODAC": 4}[x]
            ),
            [task_name],
            task_name_tabs,
        )

        # Only show charge/spin inputs for OMol task
        task_name.input(
            lambda x: (
                gr.Number(visible=True) if x == "OMol" else gr.Number(visible=False)
            ),
            [task_name],
            total_charge,
        )
        task_name.input(
            lambda x: (
                gr.Number(visible=True) if x == "OMol" else gr.Number(visible=False)
            ),
            [task_name],
            spin_multiplicity,
        )

    demo.launch()


if __name__ == "__main__":
    # On load, build and install the gradio_molecul3d fork
    subprocess.call(
        ["gradio", "cc", "install"], cwd=Path(__file__).parent / "gradio_molecule3d/"
    )
    subprocess.call(
        ["gradio", "cc", "build"], cwd=Path(__file__).parent / "gradio_molecule3d/"
    )
    subprocess.call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            Path(__file__).parent
            / "gradio_molecule3d/"
            / "dist/gradio_molecule3d-0.0.7-py3-none-any.whl",
        ],
        cwd=Path(__file__).parent.parent,
    )

    os.makedirs("/data/custom_inputs", exist_ok=True)

    # Load gradio_molecule3d only once it's built and installed
    from gradio_molecule3d import Molecule3D

    main()
