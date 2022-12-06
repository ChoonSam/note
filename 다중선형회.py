import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/Users/82102/OneDrive/바탕 화면/최종과 최초/공사일보(z-0001)CSV.csv")
X = dataset.iloc[:,[1,3,4,5,6,7,8]].values
y = dataset.iloc[:,[9]].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [0,6])], remainder='passthrough')
X = ct.fit_transform(X)
# 원 핫 인코딩 이후

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

reg = LinearRegression()
reg.fit(X_train, y_train)

#plt.scatter(X, y, color='blue') # 산점도
#plt.plot(X, reg.predict(X), color='green') # 선 그래프
#plt.title('Score by hours (genius)') # 제목
#plt.xlabel('hours') # X 축 이름
#plt.ylabel('score') # Y 축 이름
#plt.show()

y_pred = reg.predict(X_test)
#print(y_pred)

reg.coef_

reg.intercept_