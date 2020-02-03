import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from numpy.random import seed
from tensorflow import set_random_seed



import csv
# Importing SKLearn clssifiers and libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
#from sklearn.cross_validation import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#from sklearn.model_selection import GridSearchCV

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
#from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold

# Load data as Pandas dataframe
train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')

#print(train.dtypes)

# feat_list = list(train.columns.values)
# for feat in feat_list:
#     print(feat,": ",sum(pd.isnull(train[feat])))


#Estrategia descarte de features irrelevantes
train = train.drop('Name', axis=1,)
train = train.drop('Ticket', axis=1,)
train = train.drop('Fare', axis=1,)
train = train.drop('Cabin', axis=1,)
train = train.drop('PassengerId', axis=1,)
#print(train.dtypes)

#preenchimento dos valores nulos da feature "age" com a mediana 
train["Age"] = train["Age"].fillna(train["Age"].median())


#print(train["Embarked"].mode())

#Preenchimento de valores nulos da feature "Embarked" com "S" (maior ocorrência)
train["Embarked"] = train["Embarked"].fillna("S")


feat_list = list(train.columns.values)
for feat in feat_list:
    print(feat,": ",sum(pd.isnull(train[feat])))

# sns.countplot(x='Pclass', data=train, palette='hls', hue='Survived')
# plt.xticks(rotation=45)
# plt.show()


#Print dos histogramas das features selecionadas em relação ao "Survived"
feature_hist = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]
 
# for each in feature_hist:
#     sns.countplot(x=each, data=train, palette='hls', hue='Survived')
#     plt.xticks(rotation=45)
#     plt.figure()
#     plt.show(block=False)
# plt.show()

#convertendo features "embarket" e "sex" em float.
train["Embarked"][train["Embarked"] == "S"] = 1.0
train["Embarked"][train["Embarked"] == "C"] = 2.0
train["Embarked"][train["Embarked"] == "Q"] = 3.0
train["Sex"][train["Sex"] == "male"] = 1.0
train["Sex"][train["Sex"] == "female"] = 2.0

# for each in train["Embarked"]:
# 	train["embarked"]=each.astype("float64")

#print(train.dtypes)

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']


#Aplicação da knn (neste exemplo, k=1 e distancia manhattan)
knn = KNeighborsClassifier(n_neighbors=1, p=1) #1=manhattan e 2=euclidian
# knn.fit(X_train, y_train) 
# print(knn.predict([[1, 1.0, 0.5, 2, 3, 3.0]]))

#k-fold cross validation
cv_results = cross_validate(knn, X_train, y_train, cv=3)  

print(cv_results)


#Rede Neural

def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):
    
    # set random seed for reproducibility
    seed(42)
    set_random_seed(42) 
    model = Sequential()
   
    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    
    # create additional hidden layers
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))
    
    # add dropout, default is none
    model.add(Dropout(dr))
    
    # create output layer
    model.add(Dense(1, activation='sigmoid'))  # output layer
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

model = create_model()
print(model.summary())

training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.33, verbose=0)
val_acc = np.mean(training.history['val_acc'])
print("\n%s: %.2f%%" % ('val_acc', val_acc*100))

