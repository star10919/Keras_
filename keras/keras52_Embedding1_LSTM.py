from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

### Embedding

# 1. 데이터
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요','추천하고 싶은 영화입니다.','한 번 더 보고 싶네요','글쎄요','별로에요','생각보다 지루해요',
        '연기가 어색해요','재미없어요','너무 재미없다','참 재밌네요','청순이가 잘 생기긴 했어요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)        # 단어종류 28개
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '청순이가': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x)        # [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

from tensorflow.keras.preprocessing.sequence import pad_sequences       # 리스트 내 크기가 각각 다를 때 패딩으로 채워줌 / 리스트 내에서는 크기가 달라도 됨
pad_x = pad_sequences(x, padding='pre', maxlen=5)       # pre : 앞/ post : 뒤       # maxlen의 크기보다 길면 maxlen의 크기에 맞춰서 잘리는데 무조건 앞이 잘림!!(한국말은 끝까지 들어라~~뒤가중요하니까)
print(pad_x)
print(pad_x.shape)      # (13, 5)

word_size = len(token.word_index)
print(word_size)        # 27
print(np.unique(pad_x))


# 원핫인코딩 하면? (13, 5) -> (13, 5, 27)         # 열(맨 뒤)은 label의 개수


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM      # Embedding 은 원핫인코더 안 한 상태에서 들어감     # Embedding : 원핫인코더&벡터화 한번에 해줌

                                                 # 인풋은 (13, 5)
model = Sequential()               # DNN에서 unit과 동일, CNN에서 filter와 동일(아무숫자나 쓸 수 있음)
model.add(Embedding(input_dim=28, output_dim=77, input_length=5))        # Embedding만 input이 맨 앞으로 감
                   # 단어사전의 개수, 라벨 개수     # 단어수, 길이
# model.add(Embedding(27, 77))
# model.add(Embedding(27, 77, input_length=5))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 77)             2079
_________________________________________________________________
lstm (LSTM)                  (None, 32)                14080
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 16,192
Trainable params: 16,192
Non-trainable params: 0
_________________________________________________________________

  => 2079가 나오는 이유 : input_dim * output_dim         # input_length는 연산에 영향 미치지 않음(input_length은 벡터화만함)
'''


# 3. 컴파일(ES), 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)


# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print("acc :", acc)