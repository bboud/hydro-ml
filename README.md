# hydro-ml

![](images/baryon_model_image.png)

The goal of this project is to replace the need for classical hydrodynamic simulation of Quark-Gluon Plasma to infer certainly
properties by converting initial state baryon distributions into final state net proton rapidity distributions using machine learning.

## Prerequisites
### Required
- [Numpy](https://numpy.org/)
- [Pytorch](https://pytorch.org/)

### Optional
These packages are only required for generating plots, training, and other applications of Jupyter notebooks.

- [Jupyter](https://jupyter.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scipy](https://scipy.org/)

## Use
### main.py
```bash
usage: hydroml [-h] -d DATASET

Run a machine learning model to generate the final-state net proton pseudorapidity distribution give a dataset of initial-state baryon density distribution.

options:
  -h, --help                      show this help message and exit
  -d DATASET, --dataset DATASET   loads the initial-state dataset
```