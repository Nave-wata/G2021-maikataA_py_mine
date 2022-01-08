import os
import glob
import librosa
import pickle
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import sys


def get_wav(path):
    dataset = []
    melspecs = []

    for file_name in glob.glob(path):
        y, sr = librosa.load(file_name)
        dataset.append(y)

        # メルスペクトログラムの計算
        melspec = librosa.feature.melspectrogram(y, sr)
        melspec = librosa.amplitude_to_db(melspec).flatten()
        melspecs.append(melspec.astype(np.float16))

    return dataset, melspecs


# 複数データを読み込み
base_path = os.path.dirname(__file__)
train_dog = os.path.join(base_path, 'testes/train/dog/*.wav')
train_other = os.path.join(base_path, 'testes/train/other/*.wav')
test_dog = os.path.join(base_path, 'testes/testes/dog/*.wav')
test_other = os.path.join(base_path, 'testes/testes/other/*.wav')

trd_dataset, trd_melspecs = get_wav(train_dog)
tro_dataset, tro_melspecs = get_wav(train_other)
ted_dataset, ted_melspecs = get_wav(test_dog)
teo_dataset, teo_melspecs = get_wav(test_other)

trd_dataset = np.array(trd_dataset)
tro_dataset = np.array(tro_dataset)
ted_dataset = np.array(ted_dataset)
teo_dataset = np.array(teo_dataset)

# 各データの振幅の平均値
trd_mean = np.sqrt(np.mean(trd_dataset**2, axis=1))
tro_mean = np.sqrt(np.mean(tro_dataset**2, axis=1))
ted_mean = np.sqrt(np.mean(ted_dataset**2, axis=1))
teo_mean = np.sqrt(np.mean(teo_dataset**2, axis=1))

# 各データのゼロクロス数
trd_zc = np.sum(librosa.zero_crossings(trd_dataset), axis=1)
tro_zc = np.sum(librosa.zero_crossings(tro_dataset), axis=1)
ted_zc = np.sum(librosa.zero_crossings(ted_dataset), axis=1)
teo_zc = np.sum(librosa.zero_crossings(teo_dataset), axis=1)

# train_feature学習データ，test_feature予測（テスト）データ
train_size = len(trd_mean) + len(tro_mean)
test_size = len(ted_mean) + len(teo_mean)
train_feature = [0] * train_size
test_feature = [0] * test_size
tmp = []

train_mean = trd_mean
train_mean = np.append(train_mean, tro_mean)
train_zc = trd_zc
train_zc = np.append(train_zc, tro_zc)
train_melspecs = trd_melspecs
train_melspecs = np.append(train_melspecs, tro_melspecs, axis=0)

test_mean = ted_mean
test_mean = np.append(test_mean, teo_mean)
test_zc = ted_zc
test_zc = np.append(test_zc, teo_zc)
test_melspecs = ted_melspecs
test_melspecs = np.append(test_melspecs, teo_melspecs, axis=0)

train_feature = [np.append(np.append(np.append(tmp, train_mean[i]), train_zc[i]), train_melspecs[i]) for i in range(train_size)]
test_feature = [np.append(np.append(np.append(tmp, test_mean[i]), test_zc[i]), test_melspecs[i]) for i in range(test_size)]

# 正誤ラベル（本番は別ファイルに作る）
train_labels = ['dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other']


# 上記のfeatureで特徴があれば教師あり学習のアルゴリズムを使い、
# なければ教師なし学習で異常検知を狙うとよい。
# アルゴリズムによっては数値の標準化が必要。
# 機械学習：下記例は教師ありのRandomForestを用いている。
model = RandomForestClassifier(random_state=0)
model.fit(train_feature, train_labels)

# 学習データ保存
save_path = os.path.join(base_path, 'pickle/')
with open(save_path + 'train_feature.pickle', mode='wb') as f:
    pickle.dump(train_feature, f, protocol = 2)

# 予測（テスト）データ保存
with open(save_path + 'test_feature.pickle', mode='wb') as f:
    pickle.dump(test_feature, f, protocol = 2)

# 学習結果保存
with open(save_path + 'model1.pickle', mode='wb') as f:
    pickle.dump(model, f, protocol = 2)
