import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("home/Pokala/MachinLearning/Salary_Data.csv")

data.head()


real_X = (data.iloc[:,0].values).reshape(-1,1)
real_Y = (data.iloc[:,1].values).reshape(-1,1)

train_X, test_X, train_Y , test_Y = train_test_split(real_X,real_Y, train_size = 0.7 , random_state=0)

Lin = LinearRegression()
Lin.fit(train_X,train_Y)
pred_Y = Lin.predict(test_X)

# train graph
plt.scatter(train_X,train_Y,color="red")
plt.plot(train_X,Lin.predict(train_X),color="blue")
plt.title("Simple linear Regression plot")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# test graph
plt.scatter(test_X,test_Y,color="red")
plt.plot(train_X,Lin.predict(train_X),color="blue")
plt.title("Simple linear Regression plot")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
