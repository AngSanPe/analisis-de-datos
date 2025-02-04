import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

iris=pd.read_csv('p4\iris (1).csv')
emails=pd.read_csv('p4\emails.csv')

A_train,A_test,B_train,B_test=train_test_split(iris,iris['species'],test_size=.3,random_state=0)
#transformar etiquetas 
label_encoder = LabelEncoder()
etiquetas_numeros = label_encoder.fit_transform(B_train)
y=etiquetas_numeros
X=A_train.drop(['species'],axis=1)

#modelo gausian
nb_classifier = GaussianNB()




scores=[]
kf = KFold(n_splits=5, shuffle=True, random_state=0)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Entrenar el clasificador con los datos de entrenamiento
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)
    score = nb_classifier.score(X_test, y_test)
    print("Precisión en el conjunto de prueba:", score)
    scores.append(score)

print(scores)

#modelo multnomial
vectorizer = CountVectorizer()
#Q = vectorizer.fit_transform(B_train)
nb_classifierM = MultinomialNB()
kf = KFold(n_splits=5, shuffle=True, random_state=0)
puntuacion=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Entrenar el clasificador con los datos de entrenamiento
    nb_classifierM.fit(X_train, y_train)
    y_pred = nb_classifierM.predict(X_test)
    score = nb_classifierM.score(X_test, y_test)
    print("Precisión en el conjunto de prueba:", score)
    puntuacion.append(score)
    
print(puntuacion)
"""

"""