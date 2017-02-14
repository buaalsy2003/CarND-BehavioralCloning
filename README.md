# CarND-Project3-BehavioralCloning

## Introduction
This project is to perform Behavioral Cloning of human driving behavior. 3 cameras (left, center, right) are mounted in front of the car to record images while driving. Other driving parameters like steering angle, throtter, brake and speed are also recorded with each set of images. Images obtained from human driving and corresponding steering angle are used to train a deep neural network. The trained model is then used to simulate self-driving by feeding new images to the model and predicting the steering angle. This is also called Behavioral Cloning. The whole process of image data collection and self-driving testing is done through Udacity Simulator in this project.  

## Data Collecting and Processing
Udacity Simulator provides a Training mode for users to create training images. Due to my hardware limitation, I am using the training images provided by Udacity for this project. There are total 8036 sets of images from center, left and right cameras, as well as driving parameters corresponding to each image set. Below is a sample image set:

| Left image    | Center image  | Right image  |
| ------------- |:-------------:| ------------ |
|![Left] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/left.jpg) | ![Center] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/center.jpg) | ![Right] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/right.jpg)

### Data Collecting
It can be noticed that images in each set are very similar except for a minor angle change. Therefore, I am mainly using the center image. It is well known that deep neural network needs a large amount of input features/data/images for training, testing and validation. I use two ways to increase my input dataset.
  1. I mirror the center images of non-zero steering angles and added them to the input data. 
  2. Also from the non-zero steering angle image sets, I use the left and right images by applying 0.03 and -0.03 steering angle respectively. 
With the two methods, the number of input images is increased to about 20000. This is enough for network training purpose. 

### Data Processing
With the original images (320 x 160), we need two processing steps before using them as training dataset. The goal is to keep useful information as much as possible while keeping the image as small as possible. 
  1. Image Clipping: From above sample images, it can also be noticed that some portions of images are sky, tree and other things not directly related to the road. There could be considered as noise for the network model. So the first step is to clip off the unimportant part of images. Based on observation, I remove the top 60 pixels, bottom 20 pixels as well as 10 pixels from left and right. 
  2. Image Resizing: Even after clipping, the image is still relatively big (300 x 80). This will greatly affect training speed. I resize the images to smaller ones (75 x 20) while keeping the X/Y ratio. This will not change the image information much especially the lane line information with lower resolution. 
The resized image from above center image is shown below:

| Resized Center image    | 
| ----------------------- |
|![Resized-Center] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/Resized-Center.png) |

Then I normalize the image color of 0 to 255 to (-0.5 to 0.5). From the original center image, I randomly pick 10% as test set and 10% from the rest as validation set. I could have used more for test and validation but I did so because I already am short of training dataset. 

## Network Structure
This project involves image classfication/regression on camera images. For such problems, Convolutional Neural Network is a good choice of machine learning algorithms. To build a CNN, tt is often a good practice to start from an existing network with similar problems. The NVidia's paper [End to End Learning for Self-driving Cars] (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) solved a very similar problem as this project. My network structure is similar to that with some minor changes. I used only 4 convolutional layers instead of 5 in NVidia's paper. The reason is that I don't have as much data to train and more complex network may cause overfitting with less input data. I use 3 fully connected layers and also two Dropout to prevent overfitting.  
Below is my Network structure:

    Conv2D(24, 3, 3, border_mode='valid', subsample=(2,2), activation='relu', input_shape=(smallSizeX,smallSizeY,3))
    Conv2D(36, 3, 3, border_mode='valid', subsample=(1,2), activation='relu')
    Conv2D(48, 3, 3, border_mode='valid', activation='relu')
    Conv2D(64, 2, 2, border_mode='valid', activation='relu')
    Flatten()
    Dense(512)
    Dropout(.5)
    Activation('relu')
    Dense(64)
    Dropout(.2)
    Activation('relu')
    Dense(10)
    Activation('relu')
    Dense(1)

And Below is the network Output Shape:

    Layer (type)                     Output Shape          Param       Connected to
====================================================================================================

    convolution2d_1 (Convolution2D)  (None, 9, 37, 24)     672         convolution2d_input_1[0][0]
    convolution2d_3 (Convolution2D)  (None, 5, 16, 48)     15600       convolution2d_2[0][0]
    convolution2d_4 (Convolution2D)  (None, 4, 15, 64)     12352       convolution2d_3[0][0]
    flatten_1 (Flatten)              (None, 3840)          0           convolution2d_4[0][0]
    dense_1 (Dense)                  (None, 512)           1966592     flatten_1[0][0]
    dropout_1 (Dropout)              (None, 512)           0           dense_1[0][0]
    activation_1 (Activation)        (None, 512)           0           dropout_1[0][0]
    dense_2 (Dense)                  (None, 64)            32832       activation_1[0][0]
    dropout_2 (Dropout)              (None, 64)            0           dense_2[0][0]
    activation_2 (Activation)        (None, 64)            0           dropout_2[0][0]
    dense_3 (Dense)                  (None, 10)            650         activation_2[0][0]
    activation_3 (Activation)        (None, 10)            0           dense_3[0][0]
    dense_4 (Dense)                  (None, 1)             11          activation_3[0][0]

## Network characteristics
From above table, the Network structure is clearly shown. 
* Layers
  I used only 4 layers in stead of 5 layers in NVidia paper. The reason behind is I don't have as much training data. The more complex Network tends to cause overfitting without enough training data. I tried a 5-layer network and the result is not as good. 
* Dropout Layers
  It is very important to include Dropout layers to prevent overfitting. The idea is to set to zero the output of each hidden neuron with given probability. This technique provents some neurons rely on the presence of particular other neurons, which forcs the network to learn more robust features. In this project I used probability 0.5, which gives a reasonable approximation to taking the geometric information.
* Activation
  I used 'relu' as my activation function. It would allow the non-linearity into your network apparently, which is a way to prevent overfitting. On the other hand, it can accelarate the convergence of stochastic gradient descent comparing to the sigmoid functions.

## How the model was trained
The model was trained on an octa-core CPU. The actually training process is a hyper parameter fine tune process. I had to try to find a good combination of parameters to produce a relaiable and robust model to get good simulation results. Below is how I adjusted my model parameters:
* Optimizer and Learning rate: 
  I used Adam optimizer with a learning rate = 0.0001 and Mean Squared Error as loss metric. I tried to use large learning rate like 0.1 and the result was not good at all due to bad converge. So I lowered the learning rate to 0.0001, which gives me a reliable model. Further decreasing the learning rate will not improve the result. My experiments showed that the lower learning rate led to increased accuracy both in terms of mse and autonomous driving. But I didn't let learning rate go below 0.0001 since it required more training epochs to converge the model and 0.0001 is good enough for the project. (I tried SGD as well but the result is not as good.)
* Epochs:
  It took 10 minutes to finish 20 epochs of training with the final test loss 0.00988. I chose the number of epochs based on the validation loss change. By observation, the training loss kept decreasing even beyond 30 epochs but validation loss was almost decreasing during the first 20 epochs but bouncing around after that, which is a sign of overfitting due to noise in the training data. On the other hand, the trained model after 20 epochs is good enough to allow self-driving in the test track. 
* Batch size:
  I used 128 as my batch size. I started with 64 and result is OK but the vehicle was off the road once a while. The reason might be less training data will cause some noise gradients. The smaller the batch the less accurate estimate of the gradient. When I increased it to 128, the result is satisfactory. I could bump the batch size up to 256 but it would take more memory and time to train the model, which is not necessary in this project. 
  
However, I am not sure why my training accuracy and valication accuracy were not changing at all. This made me think the 0.54 testing accuracy wont give me any clue on how the real accuracy is. Asked the question on forum and several people suggested not to care much about the accuracy but only focus on the loss as a way to validate the model. 

## Simulation Results and Discussion

When the trained network is ready, it is time to test it on the simulator. 
First of all, I use the model to predict steering angles by feeding in images at real time. With the steering angles, the car can be driven autonomously. By default, we could give a constant throttle. But the simulation result using constant throttle is not good. Car would hit curb often especially at sharp turns. This mimics the real world behavior. Drivers tend to slow down at sharp turns to avoid going off the road. So in drive.py, I use a piece-wise constant throttle depending on the steering angles. The larger steering angle, the smaller throttle.
With the piecewise constant throttle function, the overall result is satisfactory in Track 1 on the simulator. The car stayed in lane for more than 20 laps. The car  wiggled and went off the lane center once a while but it found its way back to lane center shortly. Such recovery is largely achieved by leveraging the left and right camera images. 

| Self-Driving Part1  | Self-Driving Part 2 |
| ------------------- | ------------------- |
|![Simulated] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/Animation1.gif) | ![Center] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/Animation2.gif) |

