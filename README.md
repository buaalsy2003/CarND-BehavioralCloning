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

## Results and Discussion

Results

| Self-Driving Part1  | Self-Driving Part 2 |
| ------------------- | ------------------- |
|![Simulated] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/Animation1.gif) | ![Center] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/Animation2.gif) |
