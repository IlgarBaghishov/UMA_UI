---
title: FAIR Chem UMA Educational Demo
emoji: 😻
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
hf_oauth: true
hf_oauth_scopes:
  - read-repos
---

This repo houses the FAIR chemistry UMA educational demo. The UMA models are available under a FAIR Chemistry license at https://huggingface.co/facebook/UMA .

1. Try examples to see how UMA works across molecules and materials
2. Explore how the UMA tasks should be applied to different structures
3. Try the UMA model with your own structures!

This repo also uses a fork of the excellent `gradio_molecule3d` package (https://huggingface.co/spaces/simonduerr/gradio_molecule3d/tree/main) in the `gradio_molecule3d` subdirectory for visualization of structures in 3dmol.js via gradio components. There are a number of small changes needed for the demo here, primarily:
* Add support for PBC visualization
* Add support for conversion of formats unsupported by 3dmol.js via ASE
* Remove UI elements that are unnecessary for this demo
* Add multi-model PDBs as frames to enable visualization of trajectories. 