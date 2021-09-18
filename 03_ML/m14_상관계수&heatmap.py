import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from icecream import ic

### 상관계수

datasets = load_iris()
# print(datasets.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

# print(datasets.target_names)
# ['setosa' 'versicolor' 'virginica']

x = datasets.data
y = datasets.target
ic(x.shape, y.shape)        # x.shape: (150, 4), y.shape: (150,)

# df = pd.DataFrame(x, columns=datasets.featur_names)
df = pd.DataFrame(x, columns=datasets['feature_names'])     # 데이터프레임 컬럼에 피쳐명 넣기
ic(df)

# df에 Target이라는  y컬럼 추가
df['Target'] = y
ic(df)

print('========================== 상관계수 히트 맵 ==========================')
ic(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()