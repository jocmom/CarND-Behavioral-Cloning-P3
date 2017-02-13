# imports
import pickle
import glob
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, ELU
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.layers import Lambda

def model(input_shape):
    kernel_size = 3

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    # crop 16 pixels from top and bottom
    #model.add(Cropping2D(cropping=((16,16), (0,0)), input_shape=(160,320,3)))
    model.add(Convolution2D(3,1,1))
    model.add(Convolution2D(24,kernel_size,kernel_size))
    model.add(ELU())
    model.add(MaxPooling2D())
    model.add(Convolution2D(36,kernel_size,kernel_size))
    model.add(ELU())
    model.add(MaxPooling2D())
    model.add(Convolution2D(48,kernel_size,kernel_size))
    model.add(ELU())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(ELU())
    model.add(Dense(128))
    model.add(ELU())
    model.add(Dense(64))
    model.add(ELU())
    model.add(Dense(16))
    model.add(ELU())
    model.add(Dense(8))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adam', lr=1e-4)
    return model

def translate(image, tx=0, ty=0):
    '''
    Translate/Shift an image by (tx, ty)
    '''
    rows,cols = image.shape[0:2]
    # Translation matrix
    M = np.float32([[1,0,tx],[0,1,ty]])
    return cv2.warpAffine(image,M,(cols,rows))

def rotate(image, angle=0):
    '''
    Rotate image by angle in degrees
    '''
    rows,cols = image.shape[0:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return cv2.warpAffine(image,M,(cols,rows))

def scale(image, factor=1.0):
    '''
    Scale image by factor
    '''
    rows,cols = image.shape[0:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),0,factor)
    return cv2.warpAffine(image,M,(cols,rows))

def brighten(image, factor=1.0):
    '''
    Change brightness of image by factor
    '''
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image[:,:,2] = image[:,:,2] * factor
    return cv2.cvtColor(image,cv2.COLOR_HSV2RGB)

def flip(image):
    '''
    Flip image left to right
    '''
    return cv2.flip(image,1)

def blur(image, kernel=5):
    '''
    '''
    return cv2.medianBlur(image,kernel)

def sharpen(image):
    '''
    '''
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def normalize_color(image_data):
    """
    Normalize the image data with scaling to a range of [-0.5, 0.5]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    return image_data/255.0 - 0.5

def preprocess_image(image):
    return cv2.resize(image, (64,32))

def augment_image(image):
    '''
    '''
    if(np.random.randint(2) > 0):
        image = flip(image)
    image = translate(image, np.random.uniform(-5,5),np.random.uniform(-5,5))
    image = rotate(image,np.random.uniform(-5,5))
    image = scale(image,np.random.uniform(0.8,1.2))
    image = np.array(image)
    return image

def image_gen(batch_size, normalize=False, augment=True):
    '''
    generator for a random batch with original and augmented data
    :param batch_size
    '''
    df = df_cars.sample(batch_size)
    batch_images = []
    batch_steerings = []
    batch_labels = []
    cameras = ['left' ,'center', 'right']
    correction = [0.3, 0., -0.3]
    while True:
        for idx, row in df.iterrows():
            #img_center = np.asarray(plt.imread(path + row['center']))

            # select left,right,center camera randomly
            cam_idx = np.random.randint(3)
            camera = cameras[cam_idx]
            # read images, there are some wrong spaces in the file strings
            image = plt.imread(path + row[camera].replace(' ', ''))
            image = preprocess_image(image)
            if augment:
                image = augment_image(image)
            if normalize:
                image = normalize_color(image)
            steering = np.float32(row['steering'] + correction[cam_idx] )
            label = np.sign(steering)

            batch_labels.append(label)
            batch_images.append(image)
            batch_steerings.append(steering)
        yield np.array(batch_images), np.array(batch_steerings)


path = './data/'
df_cars = pandas.read_csv(path + 'driving_log.csv')
print("Number of images:", len(df_cars))


N_EPOCHS = 5
BATCH_SIZE = 256
input_shape=(32,64,3)

m = model(input_shape)
m.fit_generator(image_gen(batch_size=BATCH_SIZE), \
                samples_per_epoch=256*100, \
                nb_epoch=N_EPOCHS, \
                validation_data=image_gen(batch_size=1, augment=True), \
                nb_val_samples=4096)
m.save('model.h5')
