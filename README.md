# Image recognition

## Introduction

In this notebook, we will explore how to load images, utilize them to train neural networks, and apply transfer learning to leverage pre-trained networks before training the neural network ourselves.

## Load Dataset

We'll be using a [dataset of cat and dog photos](https://www.kaggle.com/c/dogs-vs-cats). Here are a couple example images:

![dog_cat.png](./dog_cat.png "dog_cat.png")

We'll use this dataset to train a neural network that can differentiate between cats and dogs. 

## Transfert Learning
You'll use networks trained on [ImageNet](http://www.image-net.org/) [available from torchvision](http://pytorch.org/docs/0.3.0/torchvision/models.html)

ImageNet is a massive dataset with over 1 million labeled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers. I'm not going to get into the details of convolutional networks here, but if you want to learn more about them, please [watch this](https://www.youtube.com/watch?v=2-Ol7ZB0MmU).

Once trained, these models work astonishingly well as feature detectors for images they weren't trained on. Using a pre-trained network on images not in the training set is called transfer learning. Here we'll use transfer learning to train a network that can classify our cat and dog photos with near perfect accuracy.

With `torchvision.models` you can download these pre-trained networks and use them in your applications. We'll include `models` in our imports now.

## Setup Instructions

### Repository Setup

Clone the course repository and set up the Python environment:

```bash
# Clone the required repositories
git clone https://github.com/SimonFerrand/UDACITY_Cat-Dog_CNN.git

# Navigate to the project directory
cd UDACITY_Cat-Dog_CNN

# Create and activate the conda environment
conda create --name Torch
conda activate Torch

# Install dependencies
pip install -r requirements.txt

# Add the environment to Jupyter
python -m ipykernel install --user --name Torch --display-name "Torch"
```


### Instructions
Follow the instructions in `CNN_Cat&Dog_Udacity.ipynb`.  


