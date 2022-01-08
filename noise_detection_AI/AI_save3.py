import os
import glob
import librosa
import pickle
import joblib
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators#　　            scitic-learnの全アルゴリズム。
from sklearn.model_selection import KFold#               K分割クロスバリデーション機能。
from sklearn.model_selection import cross_val_score#     クロスバリデーションのスコアを出力する機能。
import warnings#  
import sys

warnings.filterwarnings('ignore')


# data augmentation: add white noise
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))


# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))


# data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


def get_wav(path):
    dataset = []
    melspecs = []

    for file_name in glob.glob(path):
        y_1, sr = librosa.load(file_name, sr = 44100)
        print(len(y_1))
        y_1, _ = librosa.effects.trim(y_1, top_db=20)

        """
        if len(y_1) <= sample_length:
            # padding
            pad = sample_length - len(y_1)
            y_1 = np.concatenate((y_1, np.zeros(pad, dtype=np.float16)))
        else:
            # trimming
            start = random.randint(0, len(y_1) - sample_length - 1)
            y_1 = y_1[start:start + sample_length]
        """
        
        y_2 = add_white_noise(y_1)
        y_3 = shift_sound(y_1)
        y_4 = stretch_sound(y_1)

        dataset.append(y_1)
        dataset.append(y_2)
        dataset.append(y_3)
        dataset.append(y_4)

        # メルスペクトログラムの計算
        melspec_1 = librosa.feature.melspectrogram(y_1, sr)
        melspec_2 = librosa.feature.melspectrogram(y_2, sr)
        melspec_3 = librosa.feature.melspectrogram(y_3, sr)
        melspec_4 = librosa.feature.melspectrogram(y_4, sr)
        melspec_1 = librosa.amplitude_to_db(melspec_1).flatten()
        melspec_2 = librosa.amplitude_to_db(melspec_2).flatten()
        melspec_3 = librosa.amplitude_to_db(melspec_3).flatten()
        melspec_4 = librosa.amplitude_to_db(melspec_4).flatten()

        melspecs.append(melspec_1.astype(np.float16))
        melspecs.append(melspec_2.astype(np.float16))
        melspecs.append(melspec_3.astype(np.float16))
        melspecs.append(melspec_4.astype(np.float16))

    return dataset, melspecs

# 複数データを読み込み
base_path = os.path.dirname(__file__)
dog_data = os.path.join(base_path, 'data/dog/*.wav')
other_data = os.path.join(base_path, 'data/other/*.wav')

dog_dataset, dog_melspecs = get_wav(dog_data)
other_dataset, other_melspecs = get_wav(other_data)

dog_dataset = np.array(dog_dataset)
other_dataset = np.array(other_dataset)

# 各データの振幅の平均値
dog_mean = np.sqrt(np.mean(dog_dataset**2, axis=1))
other_mean = np.sqrt(np.mean(other_dataset**2, axis=1))

# 各データのゼロクロス数
dog_zc = np.sum(librosa.zero_crossings(dog_dataset), axis=1)
other_zc = np.sum(librosa.zero_crossings(other_dataset), axis=1)

# train_feature学習データ，test_feature予測（テスト）データ
data_size = len(dog_mean) + len(other_mean)
train_feature = [0] * data_size
tmp = []

train_mean = dog_mean
train_mean = np.append(train_mean, other_mean)
train_zc = dog_zc
train_zc = np.append(train_zc, other_zc)
train_melspecs = dog_melspecs
train_melspecs = np.append(train_melspecs, other_melspecs, axis=0)

train_feature = [np.append(np.append(np.append(tmp, train_mean[i]), train_zc[i]), train_melspecs[i]) for i in range(data_size)]

# 正誤ラベル（本番は別ファイルに作る）
train_labels = ['dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'
               , 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other']


train_feature = np.array(train_feature)
train_labels = np.array(train_labels)

X_train, X_test, y_train, y_test = train_test_split(train_feature, train_labels, test_size=0.2, shuffle=False)


##アルゴリズム総当たり
allAlgorithms = all_estimators(type_filter = "classifier")#　"classifier"タイプの全てのアルゴリズムを取得する。
kfold_cv = KFold(n_splits = 5, shuffle = True)#　K分割クロスバリデーション用オブジェクト。5分割。

for(name, algorithm) in allAlgorithms:#　全てのアルゴリズムで、5通りのデータの分け方で学習した場合の精度を出力。
    try:
        clf = algorithm()
        
        if hasattr(clf,"score"):
            scores = cross_val_score(clf, X_train, y_train, cv = kfold_cv)
        print(name, "の正解率")
        print(scores)

    except Exception:#　　　　　　　　　　エラーを出したアルゴリズムは無視する。
        pass

# 学習データ保存（前処理・かさ増し1）
save_path = os.path.join(base_path, 'pickle/')
with open(save_path + 'model2.joblib', mode='wb') as f:
    joblib.dump(clf, f, protocol = 2)

# ラベル保存（かさ増し1に対応）
with open(save_path + 'add_train_feature.joblib', mode='wb') as f:
    joblib.dump(train_feature, f, protocol = 2)

sys.exit()
# 学習結果保存
with open(save_path + 'add_train_labels.pickle', mode='wb') as f:
    pickle.dump(train_labels, f, protocol = 2)
