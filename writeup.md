#**Behavioral Cloning** 

##Writeup for the Behavioral Cloning Project for the Udacity Self-Driving Car Nanodegree

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_cnn]: ./examples/nvidia_cnn.png "Model Visualization"
[brighten]: ./examples/brighten.png "Brighten Image Augmentation"
[shift]: ./examples/shift.png "Shifted Image Augmentation"
[flip]: ./examples/flip.png "Norma and Flipped Image"
[image6]: ./examples/placeholder_small.png "Normal Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md this file summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

The model is based on the Nvidia convolutional neural network for self driving cars. It consists of 3 convolutional layer wite 5x5 filter and a stride length of 2 and 2 convolutional layers with 3x3 filter and a stride length of 1. The depths are ascending from 24 to 64 (model.py lines)

After the convolutional layers there are 5 fully connected layers with decreasing outputs from 1164 to 1 which outputs the steering value

The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

The weights are initialized with a normal distribution scaled by fan_in (He et al., 2014) 

#### 2. Attempts to reduce overfitting in the model

One change to the original NVIDIA net are the additional . 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like starting from left to right :

![alt text][image3]
![alt text][image4]
![alt text][image5]

I randomly shuffled the data set and put 20% of the data into a validation set. 
I didn't repeat this on track 2 because I think this track is for validation only and the learned model from track 1 should generalize on track 2

#### Flipped Images
To augment the data sat, I also flipped images and angles thinking that this would improve generalization because track 1 has more left than right corners and flipping equlize this bias. When flipping the steering value has to be inverted. For example, here is an image that has then been flipped:

![][flip]

#### Shifted Images
I also shifted the images vertically. This is to simulate ascensions/descensions which is not relevant in track 1 but in track 2. I also tested to shift images horizontally but this had no positive effect. Probably it makes senses to adapt the steering value to the horizontally shifted images
![][shift]

#### Darkened/Brightened Images
The images are also darkened and brightened to react not too sensible on different light conditions. An Improvement to this might be adding shadows to the images.
![][brighten]

After the collection process, I had around 15000 images. I then preprocessed this data by cropping 20% from the top because the sky, trees, etc. should not have an influence on the steering direction. There are also cropped 12.5% from the bottom to remove the car frame.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. There is also no augmentation on the validation set. The ideal number of epochs was 5 as evidenced by the training rate didn't improve anymore and at some point the validation rate got even worse which is a sign of overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### References
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html