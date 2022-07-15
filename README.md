# Hydro-ML

The goal of this project is to replace the need for classical hydrodynamic simulation of Quark-Gluon Plasma by converting initial state baryon distributions into final state net proton rapidity distributions using machine learning.

### Modules
This project contains a few Jupyter notebooks. The format adhears to the folowing format: 
- `{model_name}_train.ipynb`: This will contain the code required to train the model relating to the specific model. The model weights will be saved to a file that corrisponds to the model name. For example, the `baryons` model's weights will be named `baryons_model.pt`. 
- `{model_name}_inference.ipynb`: This notebook will load the model saved from the train notebook. These notebooks are for inference only and will not train the model.

### Prerequisites
Requirements for the software and other tools to build, test and push
- [Jupyter](https://jupyter.org/)
- [Pytorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)