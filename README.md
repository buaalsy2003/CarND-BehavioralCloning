# CarND-Project3-BehavioralCloning

## Introduction
This project is to perform Behavioral Cloning of human driving behavior. 3 cameras (left, center, right) are mounted in front of the car to record images while driving. Images obtained from human driving as well as other driving parameters such as Steering Angle are used to train a deep neural network. The trained model is then used to simulate self-driving (This is also called Behavioral Cloning). The whole process of image data collection and self-driving testing is done through Udacity Simulator in this project.  

## Data Processing
Udacity Simulator provides a Training mode for users to create training images. Due to my hardware limitation, I am using the training images provided by Udacity for this project. There are total 8036 sets of images from center, left and right cameras, as well as driving parameters corresponding to each image set. Below is a sample image set:
| Left image    | Center image  | Right image  |
| ------------- |:-------------:| ------------ |
|![Left] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/left.jpg) | ![Center] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/center.jpg) | ![Right] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/right.jpg)

It can be noticed that images in each set are very similar except for a minor angle change. And each set of images has a corresponding 

## Network Structure

## Results and Discussion

Results

| Self-Driving Part1  | Self-Driving Part 2 |
| ------------------- | ------------------- |
|![Simulated] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/Animation1.gif) | ![Center] (https://github.com/buaalsy2003/CarND-BehavioralCloning/blob/master/Animation2.gif) |
