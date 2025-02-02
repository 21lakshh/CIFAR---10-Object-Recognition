# CIFAR-10 Object Recognition using ResNet-50

## Overview
This project implements object recognition on the CIFAR-10 dataset using a pre-trained ResNet-50 model. CIFAR-10 consists of 60,000 images categorized into 10 different classes. The model achieves high accuracy on both training and validation sets.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.  
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Model Architecture
- **Backbone:** ResNet-50 (pre-trained on ImageNet)
- **Input Shape:** 32x32 RGB images
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy

## Results
- **Training Accuracy:** 97.28%
- **Validation Accuracy:** 93.57%
