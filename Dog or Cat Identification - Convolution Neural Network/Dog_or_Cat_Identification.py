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

classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics=['accuracy'])

#Part 2 Fitting the CNN to the input

from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8002,
        epochs=25, 
        validation_data=test_set,
        nb_val_samples = 2000)
#ImportError: Could not import PIL.Image. The use of `load_img` requires PIL.
#can be solved by pip install pilow