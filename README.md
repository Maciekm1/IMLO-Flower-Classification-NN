# Flowers102 CNN Classifier

This project contains a Convolutional Neural Network (CNN) implemented in PyTorch for classifying images from the Flowers102 dataset. Final test accuracy of around 65% on the Flowers102 test set (70% validation). The model is located in model.py.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of Miniconda

## Installation

To set up your environment to run this code, follow these steps:

1. Navigate to the repository using cd
cd repository-directory
2. Create env with required packages, this command will create a venv called torch:
conda env create -f environment.yml
3. Activate the environment:
conda activate torch


# Running the Model
IMPORTANT: Change ROOT variable to the root of Flowers102 dataset if already locally installed

## Model Inference
To run the trained model on the test dataset, follow these steps:

Ensure the trained model weights 'model_weights.pth' is in the project directory.
Ensure the ROOT variable is set correctly inside modelInference.py

Activate the Conda environment:
conda activate torch

Run the modelInference.py python file to load the model and perform inference on the test data.
python modelInference.py

the final accuracy will be printed in the console at the end.


## Model Training
To train the model from scratch using the training and validation datasets, follow these steps:
Ensure the ROOT variable is set correctly inside modelTraining.py

Activate the Conda environment:
conda activate torch

Run the modelTraining.py python file to train the model.
python modelTraining.py

Epochs will be printed at the end, along with the specific losses and accuracy. Final accuracy and evaluation metrics will be printed at the end, along with the training time. The model weights will be saved in the current dir, overriding the 'model_weights.pth' file.