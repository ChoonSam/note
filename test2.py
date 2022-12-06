import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

#dataset = pd.read_csv('C:/Users/82102/OneDrive/바탕 화면/PythonMLWorkspace(LightWeight)/ScikitLearn/PolynomialRegressionData.csv')
#X = dataset.iloc[:, :-1].values
#귀y = dataset.iloc[:, -1].values

dataset = pd.read_csv("C:/Users/82102/OneDrive/바탕 화면/최종과 최초/공사일보(z-0001)CSV.csv")
X = dataset.iloc[:,[1,3,4,5,6,7,8]].values
y = dataset.iloc[:,[9,10,11,12,13,14,15]].values

reg = LinearRegression()
reg.fit(X, y) # 전체 데이터로 학습

plt.scatter(X, y, color='blue') # 산점도
plt.plot(X, reg.predict(X), color='green') # 선 그래프
plt.title('Score by hours (genius)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

print(X)
print(y)