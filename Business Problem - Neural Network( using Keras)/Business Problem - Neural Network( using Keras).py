# Artificial Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
"""
#onehotencoder = OneHotEncoder(categories = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
Instead of that we can just use OneHotEncoder as shown below.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [3]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())


"""
#from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
import keras
from keras.models import Sequential
from keras.layers import Dense

#Intitializing the ANN

classifier = Sequential()

#Adding the first input layer and hidden layer
#classifier.add(Dense(output_dim=6,init='uniform'))
#Activation function in hidden is relu that is retifier
#input_dim = 11 inputs not required now
# units = 6 i.e. 6 nodes 
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim = 11))

# Adding Second hidden layer 

classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#Adding the output layer
# sigmoid function = sigmoid
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))


#Compiling the ANN

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Fitting the ANN into the training set
classifier.fit(X_train,y_train,batch_size =10,epochs=150)

#Part 3 . Making prediction and evaluate the model

#Predicting the Test  set results.
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5) 

#Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



