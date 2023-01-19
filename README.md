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

### Dataset Format
The dataset provided should be a binary file. The following code is responsible for importing the data.
The dataset should be a float32 formatted binary file. **The first line should be eta (x-axis)** and the corresponding 
lines should be the list of the net baryon distribution data.

```python
# Reshape the data to (Number of events, Number of data points)
data = data.reshape(data.size // gridNx, gridNx)
self.eta = data[0] # Eta (First Line)
self.data = data[1:] # (Individual net baryon distributions)
```

### Output
The output fill will be the net proton pseudorapidity distribution in the same format as the input data.
