# 수학 연산 및 시각화
import numpy as np
import matplotlib.pyplot as plt
from time import time
# Keras 및 머신러닝 관련 패키지
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix
# Keras 신경망 관련 모듈
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense, InputLayer
from keras.layers import Conv2D, MaxPool2D
import test

# 하이퍼 파라미터(학습 반복, 배치)
MY_EPOCH = 3
MY_BAATCH = 300

# 패션 MNIST 데이터셋 로드
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
print('\n학습용 입력 데이터 요약:', X_train.shape)
print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)

# 첫 번째 학습용 데이터의 픽셀 값 출력
print('학습용 데이터:\n', X_train[0])
plt.imshow(X_train[0], cmap='gray')
plt.show()
print('데이터 라벨:\n', Y_train[0])

# 입력 데이터 정규화
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN 입력을 위한 차원 추가
train = X_train.shape[0]
X_train = X_train.reshape(train, 28, 28, 1) # (60000, 28, 28, 1)
test = X_test.shape[0]
X_test = X_test.reshape(test, 28, 28, 1) # (10000, 28, 28, 1)

# 학습용 출력데이터를 10개의 클래스로 원-핫 인코딩 변환
print('원핫 인코딩 전:', Y_train[0])
Y_train = to_categorical(Y_train, 10)
print('원핫 인코딩 후:', Y_train[0])

# 평가용 출력데이터도 동일하게 