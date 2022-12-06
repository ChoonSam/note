import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("C:/Users/82102/OneDrive/바탕 화면/최종과 최초/공사일보(z-0001)CSV.csv")
X = dataset.iloc[:,[1,3,4,5,6,7,8]].values
y = dataset.iloc[:,[9,10,11,12,13,14,15]].values

lin_reg = LinearRegression()

poly_reg = PolynomialFeatures(degree=6) # 2차
X_poly = poly_reg.fit_transform(X)
X_poly[:7] # [x] -> [x^0, x^1, x^2] -> x 가 3이라면 [1, 3, 9] 으로 변환

poly_reg.get_feature_names_out()

#lin_reg = LinearRegression()
lin_reg.fit(X_poly, y) # 변환된 X 와 y 를 가지고 모델 생성 (학습)

plt.scatter(X, y, color='blue')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='green')
plt.title('Score by hours (genius)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.ylim(20220101060000,)
plt.show()

print(X_poly)
print(y)