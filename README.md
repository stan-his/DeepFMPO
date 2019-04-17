# DeepFMPO
Code accompanying the paper "Deep reinforcement learning for multiparameter optimization in de novo drug design".



## Instructions

To run the main program on the same data as described in the paper just run:
```sh
python Main.py
```
It is also possible to run the program on a custom set of lead molecules and/or fragments. 
```sh
python Main.py lead_file.smi fragment_molecules.smi
```
Molecules that are genertated during the process can be vied by running:
```sh
python Show_Epoch.py n
```
where `n` is the epoch that should be viewed. This shows two columns of molecules. The first column contains the original lead molecule, while the second column contains modified molecules.
Any global parameters can be changed by changing them in the file "Modules/global_parameters.py"

