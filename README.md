# Pancreatic Cancer Detection from CT Images Using Deep Learning (CNN)

This repository contains the MATLAB code for detecting pancreatic cancer from CT images using a Convolutional Neural Network (CNN).

## Contents
- `train_model.m`: Code to train the CNN model.
- `classify_image.m`: Code to classify a CT image using the trained model.

## Dataset
Prepare your dataset in the following structure inside a folder named `training images`:
```
training images/
├── cancer/
└── non_cancer/
```

## Authors
Sabinisha A, Nithiya K, Srika A  
Department of ECE, E.G.S. Pillay Engineering College

## Instructions
1. Place your CT image dataset in the correct folder structure.
2. Run `train_model.m` to train the network.
3. Run `classify_image.m` to classify a new image.
