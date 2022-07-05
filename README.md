# Hydro-ML

The goal of this project is to replace the need for classical hydrodynamic simulation of Quark-Gluon Plasma by converting initial state baryon distributions into final state net proton rapidity distributions using machine learning.

This project focuses on using a Deep Convolutional Generative Adversarial Network(DCGAN) to perform this operation. Modifications of this model will be performed in the future.

### Modules
In this project, you will find two different Jupyter Notebooks. The first is named 'train_model.ipynb' which is responsible for the logic that trains the model and saves it to a file. The second file is 'run_model.ipynb' and is responsible for loading the model and running it on some test data.

### Data
There are two different data loading modules in the `data.py`. `RealData` will load data from your data, and `Data` will load a gaussian distribution that resembles the data. Changing the dataset variable will change the data that is trained!

```python
dataset = RealData( n_samples )

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)
```

### Prerequisites
Requirements for the software and other tools to build, test and push
- [Jupyter](https://jupyter.org/)
- [Pytorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
