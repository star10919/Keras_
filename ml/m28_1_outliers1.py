import numpy as np
aaa = np.array([1,2,-1000, 4, 5, 6, 7, 8, 90, 100, 500])

### 이상치 처리 - 평균값으로 하지 않고, 중위값으로 함!!!!!
# 1. 삭제
# 2. Nan 처리후 -> 보간         // linear기준으로 보간 됨
# 3.             ................. (결측치 처리 방법과 유사)
# iqr = 3사분위 - 1사분위

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 :", quartile_1)
    print("중위 :", q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(aaa)

print('이상치의 위치 :', outliers_loc)

'''
1사분위 : 3.0
중위 : 6.0
3사분위 : 49.0
이상치의 위치 : (array([ 2, 10], dtype=int64),)
'''

# 시각화
# 위 데이터를 boxplot으로 그리시오!!
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()
