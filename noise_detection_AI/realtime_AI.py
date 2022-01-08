import glob
import os
import wave
import librosa
import numpy as np
import sounddevice as sd
import pickle
import sys


def get_wav(path):
    dataset = []
    melspecs = []

    for file_name in glob.glob(path):
        y, sr = librosa.load(file_name, sr=44100)
        dataset.append(y)

        # メルスペクトログラムの計算
        melspec = librosa.feature.melspectrogram(y, sr)
        melspec = librosa.amplitude_to_db(melspec).flatten()
        melspecs.append(melspec.astype(np.float16))

    return dataset, melspecs


base_path = os.path.dirname(__file__)
FILE_NAME = os.path.join(base_path, 'data/test.wav')  # 保存するファイル名
wave_length = 2  # 録音する長さ（秒）
sample_rate = 44100  # サンプリング周波数
min_vol = 0.001

# 学習結果読み込み
open_path = os.path.join(base_path, 'pickle/')
with open(open_path + 'model_main.pickle', mode='rb') as f:
    model = pickle.load(f)

while True:
    # 録音開始（wave_length秒間録音。wait で録音し終わるまで待つ）
    data = sd.rec(int(wave_length * sample_rate), sample_rate, channels=1)
    sd.wait()

    # ノーマライズ。量子化ビット16bitで録音するので int16 の範囲で最大化する
    data = data / data.max() * np.iinfo(np.int16).max

    # float -> int
    data = data.astype(np.int16)

    # ファイル保存
    with wave.open(FILE_NAME, mode='w') as wb:
        wb.setnchannels(1)  # モノラル
        wb.setsampwidth(2)  # 16bit=2byte
        wb.setframerate(sample_rate)
        wb.writeframes(data.tobytes())  # バイト列に変換

    tmp = []
    # データの前処理
    dataset, melspecs = get_wav(FILE_NAME)
    dataset = np.array(dataset)
    mean = np.sqrt(np.mean(dataset**2, axis=1))
    print(mean)
    zc = np.sum(librosa.zero_crossings(dataset), axis=1)
    test_feature = [np.append(np.append(np.append(tmp, mean[i]), zc[i]), melspecs[i]) for i in range(len(mean))]

    # 予測 & 予測結果出力
    pred = model(test_feature)
    print(f"predict_data = {pred}")
