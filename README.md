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
[left_center_right]: ./examples/left_center_right.png "Left Center Right Images"
[shift]: ./examples/shift.png "Shifted Image Augmentation"
[flip]: ./examples/flip.png "Norma and Flipped Image"
[recovery]: ./examples/recovery.png "Recovery Image"
[steering_distribution]: ./examples/steering_distribution.png "Steering Distribution"

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

The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). Also I introduced a 1x1 inception layer after normalization to let the

The weights are initialized with a normal distribution scaled by fan_in (He et al., 2014) 

#### 2. Attempts to reduce overfitting in the model

One change to the original NVIDIA net are the additional Dropout layers on the first two fully connected layers. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). I used 5 epochs and an epochs size of 12800. The mean squared error didn't improve anymore by increasing the epochs.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia Model. I thought this model might be appropriate because it was known to work. I used a smaller im age size (64x64) and therefoe I though it makes sense to use 3x3 filters from the beginning instead of the 5x5 filters. To reduce dimensionality I used MaxPooling layers.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set(20%). I found that my first model had a low mean squared error on the training and validation set, so I gave it a try on the track. For the first try it was already performing pretty good but the car didn't take the left corner after the bridge on track 1. I further finetuned the model by adding and removing layers but whatever I did the model did never work for the whole track. Therefore I went back to the original Nvidia model and without any finetuning it worked on the complete track. I added an Inception layer to let the model find the best color space and added Dropout layers to improve generalization and therefore I got better results on track 2 although the model cannot completely drive track 2.

At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

- Conv1: 1x1x3 Inception layer 
- Conv2: 5x5x3 convolutional layer, stride length of 2
- Conv3: 5x5x24 convolutional layer, stride length of 2
- Conv4: 5x5x36 convolutional layer, stride length of 2
- Conv5: 3x3x48 convolutional layer, stride length of 1
- Conv6: 3x3x64 convolutional layer, stride length of 1
- Flatten1: Flatten layer
- FC1: 1164 fully connected layer
- Dropout1: 50% dropout layer
- FC2: 100 fully connected layer
- Dropout2: 50% dropout layer
- FC3: 50 fully connected layer
- FC4: 10 fully connected layer
- FC5: 1 output layer


Here is a visualization of the architecture 

![][nvidia_cnn]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. 
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like starting from left to right :

![][recovery]

I randomly shuffled the data set and put 20% of the data into a validation set. 
I didn't repeat this on track 2 because I think this track is for validation only and the learned model from track 1 should generalize on track 2.

#### Left and Right Camera Images
There are three cameras placed in simulator car. One in the center, one left and one right. The left and right images can be used for recovery and I added 
a steering value from 0.3 to the left and -0.3 to the right image.

![][left_center_right]

Here you can see the overall steering distribution. Maybe it makes sense to remove some images from the dataset having a steering angle around 0 to reduce 
bias to this angle and ease recovery. 

![][steering_distribution]

#### Flipped Images
To augment the data sat, I also flipped images and angles thinking that this would improve generalization because track 1 has more left than right corners and flipping equlize this bias. When flipping the steering value has to be inverted. For example, here is an image that has then been flipped:

![][flip]
#### Shifted Images
I also shifted the images vertically. This is to simulate ascensions/descensions which is not relevant in track 1 but in track 2. I also tested to shift images horizontally but this had no positive effect. Probably it makes senses to adapt the steering value to the horizontally shifted images
![][shift]

#### Darkened/Brightened Images
The images are also darkened and brightened to react not too sensible on different light conditions. An Improvement to this might be adding shadows to the images.
![][brighten]

After the collection process, I had around 15000 images. I then preprocessed this data by cropping 20% from the top because the sky, trees, etc. should not have an influence on the steering direction. They are also cropped 12.5% from the bottom to remove the car frame.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. There is also no augmentation on the validation set. The ideal number of epochs was 5 as evidenced by the training rate didn't improve anymore and at some point the validation rate got even worse which is a sign of overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### References
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html