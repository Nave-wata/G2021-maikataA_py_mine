import sys
import numpy as np
import struct
import os
import time
from PyQt5 import QtWidgets, QtGui, QtCore
from module.sinwave import SinWave
from module.Form import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):
    # settings
    RATE = 44100
    CHUNK = 1024
    N = 1024
    UPDATE_SECOND = 10
    SOUND_TIME = 0  # 単音を再生する時間[s]
    base_path = os.path.dirname(__file__)

    def __init__(self, parent=None):
        # valuables
        self.starttime = 0.0
        self.flag = 0
        self.count = 0

        super(MainWindow, self).__init__(parent=parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.max_amplitudeSpectrum = np.zeros(self.CHUNK // 2)

        # タイマーセット
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.UPDATE_SECOND)

        self.sw = SinWave()
        self.LR_select = self.sw.L

    def update(self):
        # マイクインプット
        data = self.sw.input()
        # 波形データ
        wave_figure = data[0:self.N]
        # 波形時間
        wave_time = range(0, self.N)
        # 周波数データセット
        self.freqlist = np.fft.fftfreq(self.N, d=1.0 / self.RATE)
        # スペクトル強度
        x = np.fft.fft(data[0:self.N])
        amplitudeSpectrum = np.abs(x)
        amplitudeSpectrum[0] = 0
        amplitudeSpectrum[self.CHUNK - 1] = 0
        # グラフにプロットする
        self.ui.curve_wave.setData(wave_time, wave_figure)
        self.ui.curve_spectrum.setData(self.freqlist, amplitudeSpectrum)

        # 最大音量の周波数のlist番号を取得し、lcd表示
        y = np.argmax(amplitudeSpectrum)

        if amplitudeSpectrum[y] > 3:
            self.ui.lcdNumber.display(abs(int(self.freqlist[y])))
            self.ui.lcdNumber_2.display(int(amplitudeSpectrum[y]))
        else:
            self.ui.lcdNumber.display(0)
            self.ui.lcdNumber_2.display(0)

        # 測定中の処理
        if self.flag != 0:
            # 音量超過時の処理
            if max(wave_figure) >= 0.9:
                self.sw.stop(self.freqlist[self.count])
                self.ui.label_5.setText("")
                QtWidgets.QMessageBox.information(
                    None, "通知", "音量が大きすぎます。音量を下げて、もう一度測定して下さい。", QtWidgets.QMessageBox.Yes)
                self.ui.progressBar.setProperty("value", 0)
                # ボタン操作を再開
                self.ui.pushButton.clicked.connect(self.slot1)
                self.ui.pushButton_2.clicked.connect(self.slot2)

                self.flag = 0
                self.count = 0

            # 一定時間(SOUND_TIME)再生し終わったら
            if time.time() - self.starttime > self.SOUND_TIME:
                self.max_amplitudeSpectrum[self.count] = amplitudeSpectrum[y]
                self.sw.stop(self.freqlist[self.count])
                print(
                    f"{self.count}/{self.N//2}--{self.freqlist[self.count]}Hz--{amplitudeSpectrum[y]}")
                self.count = self.count + 1

                self.sw.play(self.freqlist[self.count], 1, 0, self.LR_select)
                self.starttime = time.time()
                # プログレスバーを更新
                self.ui.progressBar.setProperty(
                    "value", self.count / (self.CHUNK // 2) * 100)

                # 測定終了
                if self.CHUNK // 2 <= self.count:
                    self.sw.stop(self.freqlist[self.count])
                    if self.flag == 1:
                        self.flag = 0
                        save_path = os.path.join(self.base_path, '../../../calibrate')
                        np.save(self.saveFile(save_path, "calibrate/earphone.npy"),
                                self.max_amplitudeSpectrum)
                    else:
                        if self.LR_select == self.sw.L:
                            self.flag = 0
                            save_path = os.path.join(self.base_path, '../../../earphone')
                            np.save(self.saveFile(save_path, "earphone/L.npy"),
                                    self.max_amplitudeSpectrum)
                        else:
                            self.flag = 0
                            save_path = os.path.join(self.base_path, '../../../earphone')
                            np.save(self.saveFile(save_path, "earphone/R.npy"),
                                    self.max_amplitudeSpectrum)
                    # ボタン操作を再開
                    self.ui.pushButton.clicked.connect(self.slot1)
                    self.ui.pushButton_2.clicked.connect(self.slot2)

                    self.ui.label_5.setText("終了")

                    self.count = 0

    def saveFile(self, foldername, initfile):
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        (fileName, selectedFilter) = QtWidgets.QFileDialog.getSaveFileName(self, 'ファイルを保存',
                                                                           foldername + f'/{initfile}', "Numpyファイル(*.npy)")
        return fileName

    def start_init(self):
        # ボタン操作の無効化
        self.ui.pushButton.clicked.disconnect(self.slot1)
        self.ui.pushButton_2.clicked.disconnect(self.slot2)

        # 初期化
        self.max_amplitudeSpectrum = np.zeros(self.CHUNK // 2)
        self.starttime = time.time()

        self.ui.progressBar.setProperty("value", 0)
        self.ui.label_5.setText(f"測定中")

    # ボタンアクション

    def slot1(self):
        self.flag = 1
        self.start_init()

    def slot2(self):
        self.flag = 2
        self.start_init()

    def slot3(self):
        # プログラムの終了
        self.close()
        self.sw.close()
        sys.exit()

    def slot4(self):
        if self.flag == 0:
            self.LR_select = self.sw.L

    def slot5(self):
        if self.flag == 0:
            self.LR_select = self.sw.R

    def closeEvent(self, event):
        # 画面クローズ時の処理
        self.slot3()

"""
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
"""

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())
