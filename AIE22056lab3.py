import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt

def classifier(d):
    d["Category"]=data1["Payment (Rs)"].apply(lambda x: 'Rich' if x > 200 else 'Poor')

def purchase(d):
    featurevec=["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]
    dataA=d[featurevec]
    A=dataA.values
    row,col=A.shape
    dataC=data1[["Payment (Rs)"]]
    C=dataC.values
    rank = np.linalg.matrix_rank(A)
    pinvA= np.linalg.pinv(A)
    X=np.dot(pinvA,C)

    return A,C,row,col,rank,pinvA,X

def stock(d):
    mean = statistics.mean(d['Price'])
    var = statistics.variance(d['Price'])

    wedn = d[d['Day'] == 'Wed']['Price']
    wd = pd.to_numeric(wedn)
    if len(wd)>0:
        wmean = statistics.mean(wd)
        wprobProfit = len(wedn[wedn > 0]) / len(wedn)
        wconProb = len(wedn[wedn > 0]) / len(d[d['Day'] == 'Wed'])
    else:
        wmean=None
        wprobProfit = None
        wconProb = None

    april = d[d['Month'] == 'Apr']['Price']
    a = pd.to_numeric(april)
    if len(wd)>0:
        aprilmean = statistics.mean(a)
    else:
        aprilmean=None

    loss = len(d[d['Chg%'] < 0]) / len(d)

    plt.scatter(d['Day'], d['Chg%'])
    plt.xlabel('Day of the Week')
    plt.ylabel('Chg%')
    plt.title('Chg% vs Day of the Week')
    plt.show()

    return mean, var, wmean, wprobProfit, wconProb, aprilmean, loss

data1=pd.read_excel(r"C:\Users\Administrator\Downloads\LabSession1Data.xlsx",sheet_name='Purchase data')
A,C,row,col,rank,pinvA,X= purchase(data1)

print(A)
print("Dimentionality of A : ",col)
print("Number of vectors: ",row)
print(C)
print("Rank of A :", rank)
print("Pseudo Inverse of A:\n",pinvA)
print("X=\n ", X)
print("Cost of 1 candy: Rs.",round(X[0][0]))
print("Cost of 1 mango: Rs.",round(X[1][0]))
print("Cost of 1 milk packet: Rs.",round(X[2][0]))

data1["Category"]=data1["Payment (Rs)"].apply(lambda x: 'Rich' if x > 200 else 'Poor')
print(data1)

data1=classifier(data1)
print(data1)

data2=pd.read_excel(r"C:\Users\Administrator\Downloads\LabSession1Data.xlsx",sheet_name='IRCTC Stock Price')
mean, var, wedn, WProfit, WcondProb, april, loss = stock(data2)

print("Mean of Price data:", mean)
print("Variance of Price data:", var)
print("Sample mean of Wednesday prices:", wedn)
print("Sample mean of April prices:", april)
print("Probability of loss over the stock:", loss)
print("Probability of profit on Wednesday:", WProfit)
print("Conditional probability of profit given that today is Wednesday:", WcondProb)