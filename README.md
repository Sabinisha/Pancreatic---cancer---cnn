# Pancreatic Cancer Detection from CT Images Using Deep Learning (CNN)

This project presents a deep learning-based approach to detect pancreatic cancer from CT scan images using a custom Convolutional Neural Network (CNN) implemented in MATLAB.

## 🧠 Model Overview
- **Architecture**: Custom CNN with convolutional, pooling, dropout, and fully connected layers.
- **Tools**: MATLAB, Deep Learning Toolbox
- **Dataset**: Public CT images from Kaggle (cancer vs non-cancer)

## 📁 Folder Structure
```
pancreatic-cancer-cnn/
├── README.md
├── trained_model/
│   └── trainedPancreaticCancerNet.mat
├── sample_data/
│   ├── cancer/
│   └── non_cancer/
├── classify_image.m
└── train_model.m
```

## 📊 Features
- Preprocessing: Normalization, Resizing, Augmentation
- Performance: ~99% accuracy on validation set
- Visualization: Annotated image outputs with predicted class

## 📌 Instructions
1. Clone this repo or download the folder.
2. Open MATLAB and set path to this folder.
3. Run `train_model.m` to train the network.
4. Use `classify_image.m` to classify a test CT image.

## 🔗 Dataset Source
[Pancreas CT Dataset - Kaggle](https://www.kaggle.com/datasets)

---
This code was developed by **Sabinisha A, Nithiya K, Srika A** as part of their final year project at **E.G.S. Pillay Engineering College**.
