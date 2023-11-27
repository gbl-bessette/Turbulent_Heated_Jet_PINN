# Turbulent_Heated_Jet_PINN
This repository includes the scripts and data used for the Sci-ML model inference of velocity and pressure components of a 3D turbulent heated jet from its temperature field (refer to Matser_Thesis.pdf). Codes are listed in the Scripts directory, data are provided in the Data folder and a simulation example is stored in the Results folder. 

Simulations can be run with the main python script under PINN_main.py, which calls all other codes. The PINN model includes different optimisation methods (or algorithms) to infer velocity and pressure components of the jet flow. A first method "temp_only", only considers target temperature values in the formulation of the data loss, and another method "bc" includes additional boundary conditions for every output of the PINN.

Two other experimental algorithms are provided to solve the optimisation sequentially in the space and time domain: refer to "progressive_spatial" and "progressivve_spatial_temporal". The user is also able to specify whether the PINN starts to train from the model's initialisation or whether it loads a checkpoint before starting the simulation. Further explanation is provided in the PINN_main.py script.



