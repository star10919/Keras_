import numpy as np

### EllipticEnvelope : outliers을 데이터셋에서 찾아내겠다!

aaa = np.array([[1,   2,   3,4,   10000,6,   7,   5000,90,  100,   5000],
                [1000,2000,3,4000,5000, 6000,7000,8,   9000,10000, 1001]])

# (2, 11) -> (11, 2)
aaa = aaa.transpose()
print(aaa.shape)    #(11, 2)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.2)   # 파라미터로 contamination만 알면 됨
outliers.fit(aaa)

results = outliers.predict(aaa)

print(results)

# [ 1  1  1  1 -1  1  1 -1  1  1  1]    -1지점이 outliers지점!
              #10000     8
