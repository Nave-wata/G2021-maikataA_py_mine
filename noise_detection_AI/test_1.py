from numpy import random
import glob
import os
import wave
import librosa
import numpy as np
import sounddevice as sd
import pickle
import pyaudio
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
with open(open_path + 'model_try.pickle', mode='rb') as f:
    model = pickle.load(f)

pa = pyaudio.PyAudio()
my_stream = pa.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=44100,
                    input=True,
                    output=False,
                    frames_per_buffer=1024,
                    input_device_index=1)

my_in_data = np.array([], dtype='int16')
my_data = np.array([], dtype='int16')

while True:
    # 録音開始（wave_length秒間録音。wait で録音し終わるまで待つ）
    """
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
    """

    my_input = my_stream.read(1024, exception_on_overflow=False)
    my_in_data = np.append(my_in_data, np.frombuffer(my_input, dtype='int16'))
    my_data = np.append(my_data, my_in_data)

    if len(my_data) > 22144:
        data = np.array(my_data)
        data = data / data.max()

        start = random.randint(0, len(data) -  22144 - 1)
        data = data[start:start +  22144]

        tmp = []
        melspec = librosa.feature.melspectrogram(data, 44100)
        melspec = librosa.amplitude_to_db(melspec).flatten()
        melspec = data.astype(np.float16)
        mean = np.sqrt(np.mean(data**2))
        zc = np.sum(librosa.zero_crossings(data))
        test_feature = [np.append(np.append(np.append(tmp, mean), zc), melspec) for _ in range(1)]
        pred = model(test_feature)
        print(f"predict_data = {pred}")
        # 予測 & 予測結果出力
        
        my_data = np.array([], dtype='int16')
""""
    # データの前処理
    dataset, melspecs = get_wav(FILE_NAME)
    dataset = np.array(dataset)
    mean = np.sqrt(np.mean(dataset**2, axis=1))
    zc = np.sum(librosa.zero_crossings(dataset), axis=1)
    test_feature = [np.append(np.append(np.append(tmp, mean[i]), zc[i]), melspecs[i]) for i in range(1)]

    # 予測 & 予測結果出力
    pred = model(test_feature)
    print(f"predict_data = {pred}")
"""