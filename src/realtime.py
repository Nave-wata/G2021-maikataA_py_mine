import pyaudio
import numpy as np
import struct
import copy
import os
import sys
from PyQt5 import QtWidgets, QtCore
from sub.realtime_form import Ui_MainWindow
from sub.realtime_AI import Noise_detection_AI


class MainWindow(QtWidgets.QMainWindow):
    base_path = os.path.dirname(__file__)

    OUTPUT_INDEX = 5
    INPUT_INDEX = 1
    CALIBRATE_PATH = os.path.join(base_path, "../calibrate/earphone.npy")  # Calibrate earphone data
    LEFT_PATH = os.path.join(base_path, "../earphone/L.npy")  # left poor earphone
    RIGHT_PATH = os.path.join(base_path, "../earphone/R.npy")  # right poor earphone
    OUTPUT_FIX = 1  # change here according to sound level

    RATE = 44100  # サンプリング周波数
    OVERFLOW_LIMIT = 20480  # Inputのバッファーの閾値

    def __init__(self, parent=None):
        # 増幅量関連の定義
        self.cnt_mag = (1, 2, 6, 12, 23, 46, 93, 186, 325, 464, 512)
        self.cnt_bwmag = (1, 4, 6, 11, 23, 47, 93, 139, 139)
        self.ls_reset = []
        self.sum_vol = []
        self.sum_vol_save = []
        self.sum_vol_add = [1.0, 1.0, 1.0]
        self.resum_vol_add = [0.0, 0.0, 0.0]

        # 各増幅量間のcount1あたりの増加量
        for i in range(9):
            self.sum_vol = np.append(self.sum_vol, np.array([1 / self.cnt_bwmag[i]]), axis=0)

        # 各周波数ごとの増幅率
        for i in range(1, 9):  # 基準の周波数含む
            for j in range(1, self.cnt_bwmag[i]+1):
                self.sum_vol_save = np.append(self.sum_vol_save, np.array([self.sum_vol[i] * j]), axis=0)
            self.sum_vol_add = np.append(self.sum_vol_add, np.array(self.sum_vol_save), axis=0)
            self.sum_vol_save = copy.copy(self.ls_reset)

        for i in range(47):  # vol[9]のために残りすべてを１にする(~512)
            self.sum_vol_add = np.append(self.sum_vol_add, 1)

        for i in range(1, 9):  # 基準の周波数含む
            for j in range(self.cnt_bwmag[i]-1, -1, -1):
                self.sum_vol_save = np.append(self.sum_vol_save, np.array([self.sum_vol[i] * j]), axis=0)
            self.resum_vol_add = np.append(self.resum_vol_add, np.array(self.sum_vol_save), axis=0)
            self.sum_vol_save = copy.copy(self.ls_reset)

        for i in range(len(self.resum_vol_add)):  # 基準の周波数を0.0にする
            if(self.resum_vol_add[i] == 1.0):
                self.resum_vol_add[i] = 0.0

        # pyqtのセットアップ
        super(MainWindow, self).__init__(parent=parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # pyaudioセットアップ
        self.pa = pyaudio.PyAudio()
        self.out_stream = self.pa.open(format=pyaudio.paInt16,
                                       channels=2,
                                       rate=self.RATE,
                                       input=False,
                                       output=True,
                                       frames_per_buffer=1024,
                                       output_device_index=self.OUTPUT_INDEX)

        self.in_stream = self.pa.open(format=pyaudio.paInt16,
                                      channels=2,
                                      rate=self.RATE,
                                      input=True,
                                      output=False,
                                      frames_per_buffer=1024,
                                      input_device_index=self.INPUT_INDEX)

        self.my_stream = self.pa.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.RATE,
                                      input=True,
                                      output=False,
                                      frames_per_buffer=1024,
                                      input_device_index=self.INPUT_INDEX)

        # 測定データ(ndarray)読み込み
        self.calib = np.load(self.CALIBRATE_PATH)  # いい音イヤホン
        self.left = np.load(self.LEFT_PATH)  # 百均イヤホン右
        self.right = np.load(self.RIGHT_PATH)  # 百均イヤホン左

        self.FLAG = False  # ON/OFFのフラグ
        self.ai_flag = False
        self.hyaku_flag = False
        self.in_frames = 0
        self.out_frames = 0

        # 周波数ごとの倍率で最も大きい値を取得する
        max = np.max(self.calib)

        # maxで割って倍率を０〜１の間に収める
        self.l_mag = self.calib / max
        self.r_mag = self.calib / max

        # FFT用に測定データを加工
        self.l_mag = np.array(self.l_mag) * self.OUTPUT_FIX
        self.r_mag = np.array(self.r_mag) * self.OUTPUT_FIX
        self.l_save = copy.copy(self.l_mag)
        self.r_save = copy.copy(self.r_mag)
        self.l_mag = np.append(np.append(self.l_mag, [0]), self.l_mag[:0:-1])
        self.r_mag = np.append(np.append(self.r_mag, [0]), self.r_mag[:0:-1])

        self.in_data = np.array([], dtype='int16')
        self.my_in_data = np.array([], dtype='int16')
        self.my_data = np.array([], dtype='int16')
        self.l_out = np.zeros(256, dtype='int16')
        self.r_out = np.zeros(256, dtype='int16')

        self.two_sec = 88200  # 2sec
        self.noising = 'other'

        self.up = np.linspace(0, 1, 256)
        self.down = np.linspace(1, 0, 256)

        self.my_ai = Noise_detection_AI()
        self.slot2()

        # タイマーセット
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

    def update(self):
        # インプットからのデータ読み込み
        if self.in_stream.get_read_available() >= 1024:
            input = self.in_stream.read(1024, exception_on_overflow=False)
            myput = self.my_stream.read(1024, exception_on_overflow=False)
            self.in_data = np.append(self.in_data, np.frombuffer(input, dtype='int16'))
            self.my_data = np.append(self.my_data, np.frombuffer(myput, dtype='int16'))
            self.in_frames += 1024

        # インプットデータのフレーム数が1024を超えたら
        if self.in_frames >= 1024:

            if len(self.my_data) > self.two_sec:
                if self.ai_flag == True:
                    self.noising = self.my_ai.jdg_start(self.my_data)
                self.my_data = np.array([], dtype='int16')

            left_data = self.in_data[:2047:2]
            right_data = self.in_data[1:2048:2]

            if self.FLAG:
                if self.ai_flag == False:
                    left_data = np.fft.ifft(np.fft.fft(left_data) * self.l_mag).real.astype('int16')
                    right_data = np.fft.ifft(np.fft.fft(right_data) * self.r_mag).real.astype('int16')
                elif self.ai_flag == True and self.noising == 'other':
                    left_data = np.fft.ifft(np.fft.fft(left_data) * self.l_mag).real.astype('int16')
                    right_data = np.fft.ifft(np.fft.fft(right_data) * self.r_mag).real.astype('int16')
                else:
                    left_data = np.fft.ifft(np.fft.fft(left_data) * (self.l_mag/2)).real.astype('int16')
                    right_data = np.fft.ifft(np.fft.fft(right_data) * (self.r_mag/2)).real.astype('int16')

            self.l_out[-256:] = self.l_out[-256:] * \
                self.down + left_data[0:256] * self.up
            self.r_out[-256:] = self.r_out[-256:] * \
                self.down + right_data[0:256] * self.up
            self.l_out = np.append(self.l_out, left_data[256:])
            self.r_out = np.append(self.r_out, right_data[256:])

            self.in_data = self.in_data[1536:]
            self.in_frames -= 768
            self.out_frames += 768

        # 出力データのフレーム数が1024を超えたら
        if self.out_frames >= 1024:
            data = np.array(
                [self.l_out[0:1024], self.r_out[0:1024]]).T.flatten()
            data = data.tolist()
            data = struct.pack("h" * len(data), *data)
            self.out_stream.write(data)
            self.l_out = self.l_out[1024:]
            self.r_out = self.r_out[1024:]
            self.out_frames -= 1024

        # オーバーフロー処理
        if self.in_frames > self.OVERFLOW_LIMIT:
            self.in_frames = 0
            self.out_frames = 0
            self.in_data = np.array([], dtype='int16')
            self.l_out = np.zeros(256, dtype='int16')
            self.r_out = np.zeros(256, dtype='int16')
            print("OVER FLOW!!")

    # ボタン操作で呼び出される関数
    def slot1(self):
        if self.FLAG:
            self.FLAG = False
        else:
            self.FLAG = True

    # 増幅処理
    def slot2(self):
        vol = self.ui.value_change(None)

        self.l_mag = copy.copy(self.l_save)
        self.r_mag = copy.copy(self.r_save)

        for i in range(10):  # self.sum_vol_addの1次は9個
            if(i == 0):
                # 前半
                self.l_mag[: self.cnt_mag[i]+1
                           ] = self.l_save[: self.cnt_mag[i]+1] * vol[i]
                self.r_mag[: self.cnt_mag[i]+1
                           ] = self.r_save[: self.cnt_mag[i]+1] * vol[i]

            elif(i == 9):
                # 前半
                self.l_mag[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1
                           ] = self.l_mag[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1] + self.sum_vol_add[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1] * vol[i]
                self.r_mag[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1
                           ] = self.r_mag[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1] + self.sum_vol_add[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1] * vol[i]
                # 後半
                self.l_mag[self.cnt_mag[9]:
                           ] = self.l_mag[self.cnt_mag[9]:] + self.sum_vol_add[self.cnt_mag[9]:] * vol[9]
                self.r_mag[self.cnt_mag[9]:
                           ] = self.r_mag[self.cnt_mag[9]:] + self.sum_vol_add[self.cnt_mag[9]:] * vol[9]

            else:
                # 前半
                self.l_mag[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1
                           ] = self.l_mag[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1] + self.sum_vol_add[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1] * vol[i]
                self.r_mag[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1
                           ] = self.r_mag[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1] + self.sum_vol_add[self.cnt_mag[i-1]+1: self.cnt_mag[i]+1] * vol[i]
                # 後半
                self.l_mag[self.cnt_mag[i]+1: self.cnt_mag[i+1]
                           ] = self.l_mag[self.cnt_mag[i]+1: self.cnt_mag[i+1]] + self.l_mag[self.cnt_mag[i]+1: self.cnt_mag[i+1]] + self.resum_vol_add[self.cnt_mag[i]+1: self.cnt_mag[i+1]] * vol[i]
                self.r_mag[self.cnt_mag[i]+1: self.cnt_mag[i+1]
                           ] = self.r_mag[self.cnt_mag[i]+1: self.cnt_mag[i+1]] + self.r_mag[self.cnt_mag[i]+1: self.cnt_mag[i+1]] + self.resum_vol_add[self.cnt_mag[i]+1: self.cnt_mag[i+1]] * vol[i]

        self.l_mag = np.append(np.append(self.l_mag, [0]), self.l_mag[:0:-1])
        self.r_mag = np.append(np.append(self.r_mag, [0]), self.r_mag[:0:-1])

    def slot3(self, state):
        if (QtCore.Qt.Checked == state):
            self.ai_flag = True
        else:
            self.ai_flag = False

    def slot4(self, state):
        if (QtCore.Qt.Checked == state):
            max = np.max([self.calib / self.left, self.calib / self.right])
            self.l_mag = self.calib / self.left / max
            self.r_mag = self.calib / self.right / max
            self.l_mag = np.array(self.l_mag) * self.OUTPUT_FIX
            self.r_mag = np.array(self.r_mag) * self.OUTPUT_FIX
            self.l_save = copy.copy(self.l_mag)
            self.r_save = copy.copy(self.r_mag)
            self.l_mag = np.append(np.append(self.l_mag, [0]), self.l_mag[:0:-1])
            self.r_mag = np.append(np.append(self.r_mag, [0]), self.r_mag[:0:-1])
            self.slot2()
        else:
            max = np.max(self.calib)
            self.l_mag = self.calib / max
            self.r_mag = self.calib / max
            self.l_mag = np.array(self.l_mag) * self.OUTPUT_FIX
            self.r_mag = np.array(self.r_mag) * self.OUTPUT_FIX
            self.l_save = copy.copy(self.l_mag)
            self.r_save = copy.copy(self.r_mag)
            self.l_mag = np.append(np.append(self.l_mag, [0]), self.l_mag[:0:-1])
            self.r_mag = np.append(np.append(self.r_mag, [0]), self.r_mag[:0:-1])
            self.slot2()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])  # アプリケーションを作成
    w = MainWindow()
    w.show()
    app.exec()
