import pandas as pd 
import  numpy as np
import matplotlib.pyplot as plt

#Importing the dataset

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting =3)
#quoting =3 ignoring the quotation

#cleaning the dataset 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]
for i in range(0,1000):
    review =re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review = review.lower()
    review= review.split()
    ps = PorterStemmer()

    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)
    
    
    
#creating the bag of model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()