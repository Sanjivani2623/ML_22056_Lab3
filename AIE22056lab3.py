import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def classifier(d,features):
    features = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    X = d[features]
    y = d['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    classifier = DecisionTreeClassifier(random_state=40)
    classifier.fit(X_train, y_train)
    d['Predicted'] = classifier.predict(X)
    return d


data1=pd.read_excel(r"C:\Users\Administrator\Downloads\LabSession1Data.xlsx",sheet_name='Purchase data')
print(data1)
featurevec=["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]
dataA=data1[featurevec]
A=dataA.values
row,col=A.shape
print(A)
print("Dimentionality of A : ",col)
print("Number of vectors: ",row)
dataC=data1[["Payment (Rs)"]]
C=dataC.values
print(C.shape)

rank = np.linalg.matrix_rank(A)
print("Rank of A :", rank)

pinvA= np.linalg.pinv(A)
print("Pseudo Inverse of A:\n",pinvA)

X=np.dot(pinvA,C)
print("X=\n ", X)
print("Cost of 1 candy: Rs.",round(X[0][0]))
print("Cost of 1 mango: Rs.",round(X[1][0]))
print("Cost of 1 milk packet: Rs.",round(X[2][0]))

data1["Category"]=data1["Payment (Rs)"].apply(lambda x: 'Rich' if x > 200 else 'Poor')
print(data1)

data1=classifier(data1,featurevec)
print(data1)