#Importing the Keras Libraries

from keras.models import Sequential
#form keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
#when using tensorflow backend using inputshape("size,size,channe3)
classifier = Sequential()
#step 1 convolution
classifier.add(Conv2d(32,3,3,input_shape(64,64,3),activation = 'relu'))