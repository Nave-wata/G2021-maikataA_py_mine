import os
import glob
import librosa
import pickle
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import sys
from time import sleep


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


# 複数データを読み込み
print("パス準備")
D_base = 'D:/Users/pr0gr/RD_club/projects/G2021-maikataA_py/noise_detection_AI/ESC-50'
C_base = os.path.dirname(__file__)

audio_path = os.path.join(D_base, 'audio/1-11687-A-47.wav')
meta_path = os.path.join(D_base, 'meta/esc50.csv')
csv_file = pd.read_csv(meta_path, encoding="ms932", sep=",")

dataset = []
melspecs = []
labels = []

print("読み込み開始")

for file_path in glob.glob(audio_path):
    # ラベルを入れる
    file_name = os.path.basename(file_path)
    tmp = csv_file.query(f'filename == "{file_name}"')

    if any(x in tmp.category.values for x in ('Snoring', 'washing_machine', 'vacuum_cleaner', 'helicopter', 'chainsaw', 'engine', 'airplane')):
        labels.append(tmp.category)
        labels.append(tmp.category)
        labels.append(tmp.category)
        labels.append(tmp.category)
        print(tmp.category)
    else:
        labels.append('other')
        labels.append('other')
        labels.append('other')
        labels.append('other')
        print('other')

    sys.exit()

    y_1, sr = librosa.load(file_path)
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


print("前処理開始")

dataset = np.array(dataset)

# 各データの振幅の平均値
mean = np.sqrt(np.mean(dataset**2, axis=1))

# 各データのゼロクロス数
zc = np.sum(librosa.zero_crossings(dataset), axis=1)

# train_feature学習データ，test_feature予測（テスト）データ
train_size = len(mean)
train_feature = [0] * train_size
tmp = []
train_feature = [np.append(np.append(np.append(tmp, mean[i]), zc[i]), melspecs[i]) for i in range(train_size)]

# 学習結果保存
save_path = os.path.join(C_base, 'pickle/')
with open(save_path + 'all_train_data.pickle', mode='wb') as f:
    pickle.dump(train_feature, f, protocol=2)
# 学習結果保存
with open(save_path + 'all_labels.pickle', mode='wb') as f:
    pickle.dump(labels, f, protocol=2)


print("学習開始")

# 上記のfeatureで特徴があれば教師あり学習のアルゴリズムを使い、
# なければ教師なし学習で異常検知を狙うとよい。
# アルゴリズムによっては数値の標準化が必要。
# 機械学習：下記例は教師ありのRandomForestを用いている。
model = RandomForestClassifier(random_state=0)
model.fit(train_feature, labels)

print("保存開始")

save_path = os.path.join(C_base, 'pickle/')

# 学習データ保存
with open(save_path + 'all_train_1.pickle', mode='wb') as f:
    pickle.dump(train_feature, f, protocol=2)

# 学習結果保存
with open(save_path + 'all_model_1.pickle', mode='wb') as f:
    pickle.dump(model, f, protocol=2)
