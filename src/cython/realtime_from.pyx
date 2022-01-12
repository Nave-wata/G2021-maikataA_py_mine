# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'realtime.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1000, 500)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 500))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 500))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 80, 241, 51))

        # 追加要素
        # スライダー
        # 生成
        # 場所比率の方がいいかも
        max = 10
        min = 1
        vlu = 4
        winx = 170
        winy = 220
        sldx = 30
        sldy = 200
        winx_add = 70

        self.sld_A = QtWidgets.QSlider(self.centralwidget)
        self.sld_A.setGeometry(winx, winy, sldx, sldy)
        self.sld_A.setMaximum(max)
        self.sld_A.setMinimum(min)
        self.sld_A.setValue(vlu)
        self.sld_A.valueChanged.connect(MainWindow.slot2)

        self.sld_B = QtWidgets.QSlider(self.centralwidget)
        self.sld_B.setGeometry(winx + winx_add*1, winy, sldx, sldy)
        self.sld_B.setMaximum(max)
        self.sld_B.setMinimum(min)
        self.sld_B.setValue(vlu)
        self.sld_B.valueChanged.connect(MainWindow.slot2)

        self.sld_C = QtWidgets.QSlider(self.centralwidget)
        self.sld_C.setGeometry(winx + winx_add*2, winy, sldx, sldy)
        self.sld_C.setMaximum(max)
        self.sld_C.setMinimum(min)
        self.sld_C.setValue(vlu)
        self.sld_C.valueChanged.connect(MainWindow.slot2)

        self.sld_D = QtWidgets.QSlider(self.centralwidget)
        self.sld_D.setGeometry(winx + winx_add*3, winy, sldx, sldy)
        self.sld_D.setMaximum(max)
        self.sld_D.setMinimum(min)
        self.sld_D.setValue(vlu)
        self.sld_D.valueChanged.connect(MainWindow.slot2)

        self.sld_E = QtWidgets.QSlider(self.centralwidget)
        self.sld_E.setGeometry(winx + winx_add*4, winy, sldx, sldy)
        self.sld_E.setMaximum(max)
        self.sld_E.setMinimum(min)
        self.sld_E.setValue(vlu)
        self.sld_E.valueChanged.connect(MainWindow.slot2)

        self.sld_F = QtWidgets.QSlider(self.centralwidget)
        self.sld_F.setGeometry(winx + winx_add*5, winy, sldx, sldy)
        self.sld_F.setMaximum(max)
        self.sld_F.setMinimum(min)
        self.sld_F.setValue(vlu)
        self.sld_F.valueChanged.connect(MainWindow.slot2)

        self.sld_G = QtWidgets.QSlider(self.centralwidget)
        self.sld_G.setGeometry(winx + winx_add*6, winy, sldx, sldy)
        self.sld_G.setMaximum(max)
        self.sld_G.setMinimum(min)
        self.sld_G.setValue(vlu)
        self.sld_G.valueChanged.connect(MainWindow.slot2)

        self.sld_H = QtWidgets.QSlider(self.centralwidget)
        self.sld_H.setGeometry(winx + winx_add*7, winy, sldx, sldy)
        self.sld_H.setMaximum(max)
        self.sld_H.setMinimum(min)
        self.sld_H.setValue(vlu)
        self.sld_H.valueChanged.connect(MainWindow.slot2)

        self.sld_I = QtWidgets.QSlider(self.centralwidget)
        self.sld_I.setGeometry(winx + winx_add*8, winy, sldx, sldy)
        self.sld_I.setMaximum(max)
        self.sld_I.setMinimum(min)
        self.sld_I.setValue(vlu)
        self.sld_I.valueChanged.connect(MainWindow.slot2)

        self.sld_J = QtWidgets.QSlider(self.centralwidget)
        self.sld_J.setGeometry(winx + winx_add*9, winy, sldx, sldy)
        self.sld_J.setMaximum(max)
        self.sld_J.setMinimum(min)
        self.sld_J.setValue(vlu)
        self.sld_J.valueChanged.connect(MainWindow.slot2)

        # チェックボックス
        self.cbA = QtWidgets.QCheckBox('Noise AI', self.centralwidget)
        self.cbA.setGeometry(500, 80, 150 ,50)
        self.cbA.stateChanged.connect(MainWindow.slot3)
        self.cbB = QtWidgets.QCheckBox('百均イヤホン', self.centralwidget)
        self.cbB.setGeometry(350, 80, 150 ,50)
        self.cbB.stateChanged.connect(MainWindow.slot4)
        #self.cbC = QtWidgets.QCheckBox('Check Box', self.centralwidget)
        #self.cbC.setGeometry(100, 300, 20 ,20)

        # テキスト
        winx = 170
        winy = 430
        labelx = 30
        labely = 30

        self.label_A = QtWidgets.QLabel(self.centralwidget)
        self.label_A.setText(' 43')
        self.label_A.setGeometry(winx, winy, labelx, labely)

        self.label_B = QtWidgets.QLabel(self.centralwidget)
        self.label_B.setText(' 86')
        self.label_B.setGeometry(winx + winx_add*1, winy, labelx, labely)

        self.label_C = QtWidgets.QLabel(self.centralwidget)
        self.label_C.setText('258')
        self.label_C.setGeometry(winx + winx_add*2, winy, labelx, labely)

        self.label_D = QtWidgets.QLabel(self.centralwidget)
        self.label_D.setText('516')
        self.label_D.setGeometry(winx + winx_add*3, winy, labelx, labely)

        self.label_E = QtWidgets.QLabel(self.centralwidget)
        self.label_E.setText(' 1k')
        self.label_E.setGeometry(winx + winx_add*4, winy, labelx, labely)

        self.label_F = QtWidgets.QLabel(self.centralwidget)
        self.label_F.setText(' 2k')
        self.label_F.setGeometry(winx + winx_add*5, winy, labelx, labely)

        self.label_G = QtWidgets.QLabel(self.centralwidget)
        self.label_G.setText(' 4k')
        self.label_G.setGeometry(winx + winx_add*6, winy, labelx, labely)

        self.label_H = QtWidgets.QLabel(self.centralwidget)
        self.label_H.setText(' 8k')
        self.label_H.setGeometry(winx + winx_add*7, winy, labelx, labely)

        self.label_I = QtWidgets.QLabel(self.centralwidget)
        self.label_I.setText('14k')
        self.label_I.setGeometry(winx + winx_add*8, winy, labelx, labely)

        self.label_J = QtWidgets.QLabel(self.centralwidget)
        self.label_J.setText('20k')
        self.label_J.setGeometry(winx + winx_add*9, winy, labelx, labely)

        winx = 130
        winy = 396

        self.labelA = QtWidgets.QLabel(self.centralwidget)
        self.labelA.setText(str(min))
        self.labelA.setGeometry(winx, winy, labelx, labely)

        self.labelB = QtWidgets.QLabel(self.centralwidget)
        self.labelB.setText(str(max))
        self.labelB.setGeometry(winx, winy - 185, labelx, labely)
        ###

        font = QtGui.QFont()
        font.setPointSize(25)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("")
        self.pushButton.setAutoDefault(False)
        self.pushButton.setDefault(False)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 20, 500, 40))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 340, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.slot1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def value_change(self, *args, **kwargs):
        vol_A = self.sld_A.value()
        vol_B = self.sld_B.value()
        vol_C = self.sld_C.value()
        vol_D = self.sld_D.value()
        vol_E = self.sld_E.value()
        vol_F = self.sld_F.value()
        vol_G = self.sld_G.value()
        vol_H = self.sld_H.value()
        vol_I = self.sld_I.value()
        vol_J = self.sld_J.value()
        return vol_A, vol_B, vol_C, vol_D, vol_E, vol_F, vol_G, vol_H, vol_I, vol_J

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Sound amplification"))
        self.pushButton.setText(_translate("MainWindow", "ON/OFF"))
        self.label.setText(_translate("MainWindow", "Sample 1"))
