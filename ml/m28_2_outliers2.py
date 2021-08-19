import numpy as np

# 실습 : 다차원의 outlier가 출력되도록 함수 수정

# aaa = np.array([[1,    2,    1000, 3,    4,    6,    7,    8,  90,   100,   1000],
#                 [1000, 2000, 3,    4000, 5000, 6000, 7000, 8,  5000, 20000, 1001]])


aaa = np.array([[1,2,3,4,5,6,7,8,9,100,11,12,13,14,15,16,300,577],
                [1,2,3,4,5,6,7,7,2,10,11,12,13,14,366,-533,566,18]])



# (2, 11) -> (11, 2)

aaa = aaa.transpose()
print(aaa.shape)        # (11, 2)
def outlier(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위 : ', quartile_1) # 2.5
    print('q2 : ', q2) # 6.5
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print('lower_bound :', lower_bound)
    print('upper_bound :', upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))    # | : or
outliers_loc = outlier(aaa)
# 분위값은 데이터에서 딱 떨어지는 수치는 아니고, 위치가 된다. 그렇기 때문에 데이터에 없는 값이 나올 수도 있다.
print('이상치의 위치 : ', outliers_loc)

# 시각화
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()
# 시각화 시켰을 때 아웃라이어의 비율이 높으면 데이터 자체의 문제가 있을 가능성이 있다. 
# 위 예제에서 보면 10개 데이터 중에서 아웃라이어의 비율이 40%가 넘는다.


'''
1사분위 :  4.0
q2 :  8.5
3사분위 :  14.0
lower_bound : -11.0
upper_bound : 29.0
이상치의 위치 :  (array([ 9, 14, 15, 16, 16, 17], dtype=int64), array([0, 1, 1, 0, 1, 0], dtype=int64))
'''