# 3D printed half Heusler high entropy alloy for thermoelectric application

This repository contains data and code related to COMSOL simulation in the paper "3D printed half Heusler high entropy alloy for thermoelectric application".

To use the code and data, first clone this repository as follows.

```shell
git clone https://github.com/Shibu778/comsol_thermo_leg.git
```

Then, create a python environment with `numpy`, `matplotlib`, `pathlib` and `os` python package.

Creating a conda environment.
```shell
conda create -n thermo_comsol --python=3.11
```

Activate the conda environment.
```shell
conda activate thermo_comsol
```

Install the required package. `os` is in-built in python, so no need to install it.

```shell
pip install numpy matplotlib pathlib
```

Then, go to `plots` directory and run the `thermo_analysis.py` file.
```shell
cd plots
python thermo_analysis.py
```




