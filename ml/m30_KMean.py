### 비지도 학습 : y값이 없어도 됨
# cluster : 군집화

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

datasets = load_iris()

irisDF = pd.DataFrame(data = datasets.data, columns=datasets.feature_names)     # x 데이터만 사용
print(irisDF)   #(150, 4)


# y값이 없는 데이터에서 y값 유추 가능
kmean = KMeans(n_clusters=3, max_iter=300, random_state=66)     # n_clusters : 3개의 라벨을 뽑겠다       # max_iter : epochs와 비슷
kmean.fit(irisDF)


results = kmean.labels_
print(results)              # 클러스터링 해서 생성한 y값
print(datasets.target)      # 원래 y값

irisDF['cluster'] = kmean.labels_       # 클러스터링해서 생성한 y값
irisDF['target'] = datasets.target      # 원래 y값

print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris_results = irisDF.groupby(['target', 'cluster'])['petal length (cm)'].count()       # target, cluster로 그룹화하고 개수세기
print(iris_results)

'''
target  cluster
0       0          50
1       1          48
        2           2   *** 일치하지 않음
2       1          14   *** 일치하지 않음
        2          36   
'''