# Flowers102 CNN Classifier

This project contains a Convolutional Neural Network (CNN) implemented in PyTorch for classifying images from the Flowers102 dataset.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of Miniconda

## Installation

To set up your environment to run this code, follow these steps:

1. Navigate to the repository using cd
cd repository-directory
2. Create a new Conda environment:
conda create --name torch python=3.8
3. Activate the environment:
conda activate torch
4. Create env wiht required packages:
conda env create -f environment.yml


# Running the Model

IMPORTANT: Change ROOT variable to the root of Flowers102 dataset if already locally installed

## Model Inference
To run the trained model on the test dataset, follow these steps:

Ensure the trained model file model.pth is in the project directory.
Activate the Conda environment:
conda activate torch

Start Jupyter Notebook:
jupyter notebook

Open the model_inference.ipynb notebook.
Run the cells in the notebook to load the model and perform inference on the test data.


## Model Training
To train the model from scratch using the training and validation datasets, follow these steps:

Activate the Conda environment:
conda activate torch

Start Jupyter Notebook:
jupyter notebook

Open the model_training.ipynb notebook.
Run the cells in the notebook to train the model. Ensure you have the training and validation datasets available as described in the notebook.