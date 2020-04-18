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
classifier.add(Conv2D(32,3,3,input_shape=(64,64,3),activation = 'relu'))


# step 2 pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))


#step 3 flattening

classifier.add(Flatten())

#step 4 full connection

classifier.add(Dense(output_dim =128,activation='relu'))  

classifier.add(Dense(output_dim =1,activation='sigmoid'))  

#Compiling the CNN

classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics=['accuarcy'])

#Part 2 Fitting the CNN to the input


