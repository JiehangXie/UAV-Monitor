# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.4.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QStatusBar,
    QWidget)
import main_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1230, 673)
        MainWindow.setMinimumSize(QSize(1230, 673))
        MainWindow.setMaximumSize(QSize(1230, 673))
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.inputVideo = QLabel(self.centralwidget)
        self.inputVideo.setObjectName(u"inputVideo")
        self.inputVideo.setGeometry(QRect(22, 148, 381, 271))
        self.inputVideo.setLayoutDirection(Qt.LeftToRight)
        self.inputVideo.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.inputVideo.setFrameShape(QFrame.Box)
        self.inputVideo.setFrameShadow(QFrame.Plain)
        self.inputVideo.setAlignment(Qt.AlignCenter)
        self.inputVideo.setWordWrap(False)
        self.inputVideo.setMargin(0)
        self.inputVideo.setOpenExternalLinks(False)
        self.segVideo = QLabel(self.centralwidget)
        self.segVideo.setObjectName(u"segVideo")
        self.segVideo.setGeometry(QRect(425, 148, 385, 273))
        self.segVideo.setLayoutDirection(Qt.LeftToRight)
        self.segVideo.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.segVideo.setFrameShape(QFrame.Box)
        self.segVideo.setFrameShadow(QFrame.Plain)
        self.segVideo.setAlignment(Qt.AlignCenter)
        self.segVideo.setWordWrap(False)
        self.segVideo.setMargin(0)
        self.segVideo.setOpenExternalLinks(False)
        self.detVideo = QLabel(self.centralwidget)
        self.detVideo.setObjectName(u"detVideo")
        self.detVideo.setGeometry(QRect(830, 148, 388, 272))
        self.detVideo.setLayoutDirection(Qt.LeftToRight)
        self.detVideo.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.detVideo.setFrameShape(QFrame.Box)
        self.detVideo.setFrameShadow(QFrame.Plain)
        self.detVideo.setAlignment(Qt.AlignCenter)
        self.detVideo.setWordWrap(False)
        self.detVideo.setMargin(0)
        self.detVideo.setOpenExternalLinks(False)
        self.beginButton = QPushButton(self.centralwidget)
        self.beginButton.setObjectName(u"beginButton")
        self.beginButton.setGeometry(QRect(1100, 510, 101, 111))
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(18, 118, 121, 31))
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label.setAlignment(Qt.AlignCenter)
        self.filePathEdit = QLineEdit(self.centralwidget)
        self.filePathEdit.setObjectName(u"filePathEdit")
        self.filePathEdit.setGeometry(QRect(835, 510, 255, 31))
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(960, 470, 121, 21))
        font1 = QFont()
        font1.setPointSize(13)
        font1.setBold(True)
        self.label_4.setFont(font1)
        self.label_4.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_4.setAlignment(Qt.AlignCenter)
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(40, 510, 71, 61))
        self.label_5.setPixmap(QPixmap(u":/icon/images/tree.png"))
        self.label_5.setScaledContents(True)
        self.frame = QLineEdit(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(860, 590, 61, 31))
        self.frame.setAlignment(Qt.AlignCenter)
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(835, 595, 31, 21))
        self.label_6.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(940, 590, 161, 31))
        self.label_7.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(30, 580, 91, 21))
        self.label_10.setFont(font1)
        self.label_10.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_10.setAlignment(Qt.AlignCenter)
        self.greenrate = QLabel(self.centralwidget)
        self.greenrate.setObjectName(u"greenrate")
        self.greenrate.setGeometry(QRect(30, 610, 91, 21))
        self.greenrate.setFont(font1)
        self.greenrate.setStyleSheet(u"color: rgb(255, 255, 255)")
        self.greenrate.setAlignment(Qt.AlignCenter)
        self.label_11 = QLabel(self.centralwidget)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(140, 580, 91, 21))
        self.label_11.setFont(font1)
        self.label_11.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_11.setAlignment(Qt.AlignCenter)
        self.buildingrate = QLabel(self.centralwidget)
        self.buildingrate.setObjectName(u"buildingrate")
        self.buildingrate.setGeometry(QRect(140, 610, 91, 21))
        self.buildingrate.setFont(font1)
        self.buildingrate.setStyleSheet(u"color: rgb(255, 255, 255)")
        self.buildingrate.setAlignment(Qt.AlignCenter)
        self.label_12 = QLabel(self.centralwidget)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(255, 580, 91, 21))
        self.label_12.setFont(font1)
        self.label_12.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_12.setAlignment(Qt.AlignCenter)
        self.label_13 = QLabel(self.centralwidget)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(370, 580, 91, 21))
        self.label_13.setFont(font1)
        self.label_13.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_13.setAlignment(Qt.AlignCenter)
        self.carCount = QLabel(self.centralwidget)
        self.carCount.setObjectName(u"carCount")
        self.carCount.setGeometry(QRect(255, 610, 91, 21))
        self.carCount.setFont(font1)
        self.carCount.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.carCount.setAlignment(Qt.AlignCenter)
        self.humanCount = QLabel(self.centralwidget)
        self.humanCount.setObjectName(u"humanCount")
        self.humanCount.setGeometry(QRect(370, 610, 91, 21))
        self.humanCount.setFont(font1)
        self.humanCount.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.humanCount.setAlignment(Qt.AlignCenter)
        self.label_14 = QLabel(self.centralwidget)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(150, 510, 70, 60))
        self.label_14.setPixmap(QPixmap(u":/icon/images/\u5efa\u7b51.png"))
        self.label_14.setScaledContents(True)
        self.label_15 = QLabel(self.centralwidget)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(255, 510, 80, 80))
        self.label_15.setPixmap(QPixmap(u":/icon/images/\u5c0f\u8f66-\u79c1\u5bb6\u8f66.png"))
        self.label_15.setScaledContents(True)
        self.label_16 = QLabel(self.centralwidget)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(380, 510, 70, 60))
        self.label_16.setPixmap(QPixmap(u":/icon/images/\u884c\u4eba.png"))
        self.label_16.setScaledContents(True)
        self.label_17 = QLabel(self.centralwidget)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(1010, 70, 101, 31))
        self.label_17.setPixmap(QPixmap(u":/logo/images/logo.png"))
        self.label_17.setScaledContents(True)
        self.label_19 = QLabel(self.centralwidget)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(460, 10, 321, 71))
        font2 = QFont()
        font2.setPointSize(15)
        font2.setBold(True)
        self.label_19.setFont(font2)
        self.label_19.setStyleSheet(u"color: rgb(255, 255, 255)")
        self.label_19.setAlignment(Qt.AlignCenter)
        self.label_21 = QLabel(self.centralwidget)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(210, 0, 821, 91))
        self.label_21.setPixmap(QPixmap(u":/background/images/banner.png"))
        self.label_21.setScaledContents(True)
        self.label_22 = QLabel(self.centralwidget)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(10, -10, 1231, 691))
        self.label_22.setPixmap(QPixmap(u":/background/images/background-image.png"))
        self.label_22.setScaledContents(True)
        self.label_23 = QLabel(self.centralwidget)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(10, 110, 411, 321))
        self.label_23.setPixmap(QPixmap(u":/background/images/block.png"))
        self.label_23.setScaledContents(True)
        self.label_24 = QLabel(self.centralwidget)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setGeometry(QRect(415, 110, 411, 321))
        self.label_24.setPixmap(QPixmap(u":/background/images/block.png"))
        self.label_24.setScaledContents(True)
        self.label_25 = QLabel(self.centralwidget)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setGeometry(QRect(820, 110, 411, 321))
        self.label_25.setPixmap(QPixmap(u":/background/images/block.png"))
        self.label_25.setScaledContents(True)
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(420, 118, 121, 31))
        self.label_2.setFont(font)
        self.label_2.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_2.setAlignment(Qt.AlignCenter)
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(825, 118, 121, 31))
        self.label_3.setFont(font)
        self.label_3.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_3.setAlignment(Qt.AlignCenter)
        self.label_26 = QLabel(self.centralwidget)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setGeometry(QRect(20, 450, 801, 201))
        self.label_26.setPixmap(QPixmap(u":/background/images/block2.png"))
        self.label_26.setScaledContents(True)
        self.label_27 = QLabel(self.centralwidget)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setGeometry(QRect(820, 450, 401, 201))
        self.label_27.setPixmap(QPixmap(u":/background/images/block2.png"))
        self.label_27.setScaledContents(True)
        self.label_28 = QLabel(self.centralwidget)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setGeometry(QRect(490, 510, 70, 60))
        self.label_28.setPixmap(QPixmap(u":/icon/images/elephant.png"))
        self.label_28.setScaledContents(True)
        self.label_29 = QLabel(self.centralwidget)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setGeometry(QRect(480, 580, 91, 21))
        self.label_29.setFont(font1)
        self.label_29.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_29.setAlignment(Qt.AlignCenter)
        self.label_30 = QLabel(self.centralwidget)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setGeometry(QRect(600, 510, 70, 60))
        self.label_30.setPixmap(QPixmap(u":/icon/images/biker.png"))
        self.label_30.setScaledContents(True)
        self.label_31 = QLabel(self.centralwidget)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setGeometry(QRect(700, 510, 70, 60))
        self.label_31.setPixmap(QPixmap(u":/icon/images/rubbish_b.png"))
        self.label_31.setScaledContents(True)
        self.label_33 = QLabel(self.centralwidget)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setGeometry(QRect(585, 580, 100, 21))
        self.label_33.setFont(font1)
        self.label_33.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_33.setAlignment(Qt.AlignCenter)
        self.label_35 = QLabel(self.centralwidget)
        self.label_35.setObjectName(u"label_35")
        self.label_35.setGeometry(QRect(690, 580, 91, 21))
        self.label_35.setFont(font1)
        self.label_35.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_35.setAlignment(Qt.AlignCenter)
        self.animalCount = QLabel(self.centralwidget)
        self.animalCount.setObjectName(u"animalCount")
        self.animalCount.setGeometry(QRect(480, 610, 91, 21))
        self.animalCount.setFont(font1)
        self.animalCount.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.animalCount.setAlignment(Qt.AlignCenter)
        self.bikeCount = QLabel(self.centralwidget)
        self.bikeCount.setObjectName(u"bikeCount")
        self.bikeCount.setGeometry(QRect(590, 610, 91, 21))
        self.bikeCount.setFont(font1)
        self.bikeCount.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.bikeCount.setAlignment(Qt.AlignCenter)
        self.rubbishCount = QLabel(self.centralwidget)
        self.rubbishCount.setObjectName(u"rubbishCount")
        self.rubbishCount.setGeometry(QRect(690, 610, 91, 21))
        self.rubbishCount.setFont(font1)
        self.rubbishCount.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.rubbishCount.setAlignment(Qt.AlignCenter)
        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(835, 560, 150, 21))
        self.label_9.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.threshold = QLineEdit(self.centralwidget)
        self.threshold.setObjectName(u"threshold")
        self.threshold.setGeometry(QRect(980, 555, 111, 31))
        self.threshold.setAlignment(Qt.AlignCenter)
        self.label_32 = QLabel(self.centralwidget)
        self.label_32.setObjectName(u"label_32")
        self.label_32.setGeometry(QRect(350, 470, 121, 21))
        self.label_32.setFont(font1)
        self.label_32.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_32.setAlignment(Qt.AlignCenter)
        self.label_18 = QLabel(self.centralwidget)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(1120, 75, 101, 21))
        self.label_18.setPixmap(QPixmap(u":/logo/images/openvino.png"))
        self.label_18.setScaledContents(True)
        self.label_34 = QLabel(self.centralwidget)
        self.label_34.setObjectName(u"label_34")
        self.label_34.setGeometry(QRect(-1, -10, 1231, 691))
        self.label_34.setPixmap(QPixmap(u":/background/images/background-image.png"))
        self.label_34.setScaledContents(True)
        MainWindow.setCentralWidget(self.centralwidget)
        self.label_34.raise_()
        self.label_22.raise_()
        self.label_27.raise_()
        self.label_26.raise_()
        self.label_25.raise_()
        self.label_24.raise_()
        self.label_23.raise_()
        self.label_21.raise_()
        self.inputVideo.raise_()
        self.segVideo.raise_()
        self.detVideo.raise_()
        self.beginButton.raise_()
        self.label.raise_()
        self.filePathEdit.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.frame.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_10.raise_()
        self.greenrate.raise_()
        self.label_11.raise_()
        self.buildingrate.raise_()
        self.label_12.raise_()
        self.label_13.raise_()
        self.carCount.raise_()
        self.humanCount.raise_()
        self.label_14.raise_()
        self.label_15.raise_()
        self.label_16.raise_()
        self.label_17.raise_()
        self.label_19.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.label_28.raise_()
        self.label_29.raise_()
        self.label_30.raise_()
        self.label_31.raise_()
        self.label_33.raise_()
        self.label_35.raise_()
        self.animalCount.raise_()
        self.bikeCount.raise_()
        self.rubbishCount.raise_()
        self.label_9.raise_()
        self.threshold.raise_()
        self.label_32.raise_()
        self.label_18.raise_()
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u57fa\u4e8ePaddlePaddle\u7684\u4f4e\u7a7a\u65e0\u4eba\u673a\u667a\u80fd\u5de1\u68c0\u7cfb\u7edf", None))
        self.inputVideo.setText(QCoreApplication.translate("MainWindow", u"\u7b49\u5f85\u89c6\u9891\u4fe1\u53f7\u8f93\u5165\u2026\u2026", None))
        self.segVideo.setText(QCoreApplication.translate("MainWindow", u"\u7b49\u5f85\u89c6\u9891\u4fe1\u53f7\u8f93\u5165\u2026\u2026", None))
        self.detVideo.setText(QCoreApplication.translate("MainWindow", u"\u7b49\u5f85\u89c6\u9891\u4fe1\u53f7\u8f93\u5165\u2026\u2026", None))
        self.beginButton.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb\u68c0\u6d4b", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u8f93\u5165\u56fe\u50cf", None))
        self.filePathEdit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u8f93\u5165\u672c\u5730\u822a\u62cd\u89c6\u9891\u6216\u65e0\u4eba\u673artmp\u6e90\u5730\u5740", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u64cd\u4f5c\u533a", None))
        self.label_5.setText("")
        self.frame.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"\u6bcf", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"\u5e27\u6267\u884c\u4e00\u6b21\u8bed\u4e49\u5206\u5272\u548c\u68c0\u6d4b*", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"\u7eff\u5730\u7387", None))
        self.greenrate.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"\u5efa\u7b51\u7387", None))
        self.buildingrate.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"\u8f66\u8f86\u6570\u91cf", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"\u884c\u4eba\u6570\u91cf", None))
        self.carCount.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.humanCount.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_14.setText("")
        self.label_15.setText("")
        self.label_16.setText("")
        self.label_17.setText("")
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"\u4f4e\u7a7a\u65e0\u4eba\u673a\u667a\u80fd\u5de1\u68c0\u7cfb\u7edf", None))
        self.label_21.setText("")
        self.label_22.setText("")
        self.label_23.setText("")
        self.label_24.setText("")
        self.label_25.setText("")
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u5730\u7269\u8981\u7d20\u5206\u5272", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u5730\u7269\u8981\u7d20\u68c0\u6d4b", None))
        self.label_26.setText("")
        self.label_27.setText("")
        self.label_28.setText("")
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"\u52a8\u7269\u6570\u91cf", None))
        self.label_30.setText("")
        self.label_31.setText("")
        self.label_33.setText(QCoreApplication.translate("MainWindow", u"\u5355\u8f66\u6216\u7535\u52a8\u8f66", None))
        self.label_35.setText(QCoreApplication.translate("MainWindow", u"\u5783\u573e\u6570\u91cf", None))
        self.animalCount.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.bikeCount.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.rubbishCount.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"\u76ee\u6807\u68c0\u6d4b\u7ed3\u679c\u7f6e\u4fe1\u5ea6\u9608\u503c\uff1a", None))
        self.threshold.setText(QCoreApplication.translate("MainWindow", u"0.2", None))
        self.label_32.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u533a", None))
        self.label_18.setText("")
        self.label_34.setText("")
    # retranslateUi

