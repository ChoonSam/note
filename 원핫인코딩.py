import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# csv 파일 불러오기
dataset = pd.read_csv('C:/Users/82102/OneDrive/바탕 화면/ㅎ. 2022년 하계현장실습생 파일 (2)/데이터셋1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 데이터 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# 학습
reg = LinearRegression()
reg.fit(X_train, y_train)

# 모델 평가
reg.score(X_train, y_train) # 학습 데이터
reg.score(X_test, y_test) # 테스트 데이터

# 테스트(R2)
y_pred = reg.predict(X_test)
r2_score(y_test, y_pred)
print(r2_score(X_test, y_pred))