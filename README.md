# **Object Recognition**  

## **Overview**  
This project implements object recognition on the CIFAR-10 dataset using a ResNet-50 deep learning model. The CIFAR-10 dataset contains 60,000 images across 10 classes, with 50,000 images for training and 10,000 for testing. The model is trained and fine-tuned for accurate classification of the dataset's objects.

### **Features**  

- ResNet-50 Architecture: A powerful deep learning model designed for image classification
- Transfer Learning: Pretrained weights on ImageNet are used for faster and more efficient training.
- Data Augmentation: Enhances dataset variability to improve model generalization.
- Metrics: Achieves high accuracy and precision across all 10 classes.

Dataset Features:

The CIFAR-10 dataset consists of 60,000 32x32 RGB images across 10 classes:  

Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.  
Size:  
Training: 50,000 images.  
Testing: 10,000 images.  

# Model Architecture

Convolutional based  
convolutional_base = ResNet50(weights='imagenet',include_top=False,input_shape=(256,256,3))  
convolutional_base.summary()  
imagenet - image data contains millions of images containing 1000's of classes  
include_top = False , it will import resnet model but drop the output layer as we only need to predict for 10 classes  

model = Sequential()  
RestNet 50 takes images for (input 256,256) so everytime we use upsampling2d the width and height weill be multiplied by 2 basically scaling up your image  
model.add(layers.UpSampling2D((2,2))) #(64,64)  
model.add(layers.UpSampling2D((2,2))) # (128,128)  
model.add(layers.UpSampling2D((2,2))) # (256,256)  
model.add(convolutional_base) # ResNet 50 model  
model.add(Flatten()) # converts matrix into 1d array  
model.add(BatchNormalization()) # to make sure all values are in same range so training process in efficient  
model.add(Dense(128,activation='relu'))  
model.add(Dropout(0.5)) # to prevent overfitting , it will turn off few neurons  
model.add(BatchNormalization()) # normalization happens after every layer  
model.add(Dense(64,activation='relu'))  
model.add(Dropout(0.5))  
model.add(BatchNormalization())  
model.add(Dense(10,activation='softmax'))  

# Model Evaluation:
- Accuracy on Training Data: 97.28%  
- Accuracy on Validation Data: 93.57%  
- Accuracy on Testing Data: 93.81%

##### **Getting Started** 
Clone the repo and install dependencies.
