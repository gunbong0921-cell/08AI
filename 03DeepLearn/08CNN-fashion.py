'''
CNN(Convolutional Neural Network : 합성곱 신경망)
  : 이미지나 영상 같은 2차원 데이터를 처리하기에 특화된 딥러닝 모델.
  주로 이미지 분류, 객체인식, 얼굴인식 등에 사용된다. 
'''
# 수학 연산 및 시각화
import numpy as np
import matplotlib.pyplot as plt
from time import time
# Keras 및 머신러닝 관련 패키지
from keras.datasets import fashion_mnist
from tensorflow.keras import utils
from sklearn.metrics import f1_score, confusion_matrix
# Keras 신경망 관련 모듈
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense, InputLayer
from keras.layers import Conv2D, MaxPool2D
import sys # exit()함수 사용 

# 하이퍼 파라미터(학습 반복, 배치)
MY_EPOCH = 3
MY_BATCH = 300

# 패션 MNIST 데이터셋 로드
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
# 학습용과 평가용 데이터를 분리해서 ndarray 형식으로 가져옴. 
print('\n학습용 입력 데이터 요약:', X_train.shape)
print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)

print('학습용 데이터:\n', X_train[0])
# 첫 번째 학습용 데이터의 픽셀 값 출력
plt.imshow(X_train[0], cmap='gray')
# 이미지를 확인해보면 '앵클부츠'가 출력된다.
plt.show()
print('데이터 라벨:\n', Y_train[0])
# sys.exit(0)

# 입력 데이터 정규화
# 이미지의 색을 표현한 데이터이므로 255로 나누어 Min-Max 정규화 
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN 입력을 위한 차원 추가
# 학습용 데이터의 샘플 수(예:60000)을 가져옴 
train = X_train.shape[0]
# CNN이 요구하는 4차원 입력 형태로 변환(샘플수, 높이, 너비, 채널수) 
# MNIST는 흑백이미지 이므로 채널수는 1이된다. 
X_train = X_train.reshape(train, 28, 28, 1) # (60000, 28, 28, 1)
# 테스트용 데이터의 샘플수(10000)을 가져와서 4차원 배열 형태로 변환 
test = X_test.shape[0]
X_test = X_test.reshape(test, 28, 28, 1) # (10000, 28, 28, 1)
# 샘플 수 확인하기 
print('샘플수', train, test)
# sys.exit(0)

# 학습용 출력데이터를 10개의 클래스로 원-핫 인코딩 변환
print('원핫 인코딩 전:', Y_train[0])
Y_train = utils.to_categorical(Y_train, 10)
print('원핫 인코딩 후:', Y_train[0])

# 평가용 출력데이터도 동일하게 원핫 인코딩 변환 
Y_test = utils.to_categorical(Y_test, 10)
print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)
# sys.exit(0)

# CNN 인공 신경망 구현 : 층(Layer)을 순차적으로 쌓는 기본적인 신경망 구조 
model = Sequential()
# 입력 데이터의 형태 지정 (입력층 역할) 
'''
흑백이미지 (28*28)에 채널수 1로 지정. CNN은 4차원 입력을 요구하므로
샘플수, 높이, 너비, 채널수로 지정해야한다. InputLayer는 실제로 뉴런이
있는 층은 아니고, 이후 Conv2D 층의 입력 형태를 지정한다. 
'''
model.add(InputLayer(input_shape=(28, 28, 1)))
# 첫 번째 합성곱 블록 (은닉층 1)
'''
Conv2D : 32개의 2*2 필터를 적용하여 특징 추출. 패딩은 same(새임)으로
지정하여 입력과 출력의 크기를 동일하게 유지. 활성화 함수는 ReLU를 적용하여
음수는 제거하고, 양수는 그대로 출력해 비선형성을 부여한다. 
'''
model.add(Conv2D(32, kernel_size=2, padding='same', activation='relu'))
'''
MaxPool2D : 풀링층으로 2*2 영역에서 최대값만 추출하여 공간 정보를
줄임. 특징요약 및 연산량감소 효과가 있다. 
'''
model.add(MaxPool2D(pool_size=2))
# 두 번째  합성곱 블록 (은닉층 2)
'''
필터의 갯수를 64개로 늘려서 더 복잡한 특징을 추출하게된다. 
'''
model.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2)) 
# 완전 연결층 (출력층으로 가기 전 변환 과정)
'''
출력층으로 가기 전 변환과정을 거치게 되는데, 2차원 이미지를 1차원
벡터로 변환해서 Dense층에 연결이 가능하게 한다. 
'''
model.add(Flatten())
# 완전연결층(은닉층)
model.add(Dense(128, activation='relu'))
# 출력층. 패션Mnist는 10가지의 아이템이므로 이와같이 설정한다. 
model.add(Dense(10, activation='softmax'))
print('\nCNN 요약')
# 모델 구조 요약 ; 층의이름, 출력크기, 파라미터수 등을 확인할 수 있다. 
model.summary()
# sys.exit(0)

# 인공 신경망 학습
# 옵티마이저 Adam, 손실함수 크로스 앤트로피, 정호가도 평가 지표 추가 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# 학습시간 계산 
begin = time()
# 인공신경망 학습 수행 
model.fit(X_train, Y_train, epochs=MY_EPOCH, batch_size=MY_BATCH, verbose=1)
end = time()
print('총 학습 시간: {:.1f}초'.format(end - begin))

# 인공 신경망 평가(첫번째 반환값 손실률은 사용하지 않음)  
_, score = model.evaluate(X_test, Y_test, verbose=1)
print('최종 정확도: {:.2f}%'.format(score * 100))

# 모델을 사용하여 테스트 데이터 예측 수행 
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
truth = np.argmax(Y_test, axis=1)

# 혼동 행렬 출력 (예측값과 실제값 비교) 
print('\n혼동 행렬')
print(confusion_matrix(truth, pred))

# F1 점수 계산 및 출력 
f1 = f1_score(truth, pred, average='micro') 
print("\nF1 점수: {:.3f}".format(f1))


