from tensorflow.keras.preprocessing.text import Tokenizer       # 자연어 전처리 : 문자를 수치화 하겠다.
import numpy as np

### 자연어 처리
'''
텍스트 데이터 학습을 위해 텍스트를 연산 가능한 정수로 변환
Tokenizer : 빈도 많은 단어 우선, 이후 순차적으로 단어 정수로 변환
-> 카테고리컬 분류처럼 토큰값 사이의 상관관계 제거 필요
'''

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])      # 텍스트를 훈련하겠다. / 리스트 형태로(여러 문장도 가능하단 소리!)

print(token.word_index)     # {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}

x = token.texts_to_sequences([text])        # 수치화 형태로 변환
print(x)        # [[3, 1, 4, 5, 6, 1, 2, 2, 7]]

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size)        # 7

x= to_categorical(x)       # 0부터 채워지니까
print(x.shape)      # (1, 9, 8)