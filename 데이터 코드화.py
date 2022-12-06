import sys
import pandas as pd

mr = pd.read_csv('C:/Users/82102/OneDrive/바탕 화면/데이터셋(다듬기)CSV.csv', encoding='cp949')

# 2. 데이터 내부의 기호를 숫자로 변환
label = []
data = []
attr_list = []

# iterrows : 행에 반복적으로 접근하면서 값을 조작할 때 사용
# enumerate와 같은 개념으로 index랑 각각의 열을 뽑아줌
for row_index, row in mr.iterrows():
    # 라벨(독 여부) 생성
    label.append(row.loc[0])
    row_data = []

    # 나머지를 데이터로
    # loc :dataframe 내 해당 row나 column을 찾을 때 사용
    for v in row.loc[1:]:
        # ord() : 특정한 한 문자를 아스키 코드 값으로 변환
        row_data.append(ord(v))

    data.append(row_data)



sys.stdout = open('test.txt', 'w')

print(data.head(0))

sys.stdout.close()