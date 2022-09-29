import cv2
import sys
from main_ui import Ui_MainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QImage, QPixmap
from threading import Thread
from inferEngine import SegModel, DetModel
import logging
import time

# 初始化日志
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.filePathEdit.setText("D:\\PythonProject\\UAV\\2.mp4")
        self.ui.beginButton.clicked.connect(self.playBtn)

    def playBtn(self):
        # 获取配置信息
        self.inputVideoPath = self.ui.filePathEdit.text() # 文件路径
        self.device = self.ui.deviceBox.currentIndex() # 设备信息
        self.frameRate = int(self.ui.frame.text()) # 帧数

        # 初始化模型
        logger.info("Model Init.")

        self.SegObject = SegModel(self.device)
        self.DetObject = DetModel(self.device)

        self.inputvideoCapture = cv2.VideoCapture(self.inputVideoPath)
        self.fps = self.inputvideoCapture.get(cv2.CAP_PROP_FPS)
        inputVideoThread = Thread(target=self.displayThread)
        inputVideoThread.start()

    def displayThread(self):
        i = 0
        while self.inputvideoCapture.isOpened():
            self.ret, self.frame = self.inputvideoCapture.read()
            if self.ret:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)
                self.ui.inputVideo.setPixmap(QPixmap.fromImage(image))
                self.ui.inputVideo.setScaledContents(True)
                cv2.waitKey(1)

                # 每N帧处理分割检测一次，防止使用GPU时显存不足
                if i % self.frameRate == 0:
                    # 语义分割线程
                    self.segVideoThread = Thread(target=self.segThread)
                    self.segVideoThread.start()
                        
                    # 目标检测线程
                    self.detVideoThread = Thread(target=self.detThread)
                    self.detVideoThread.start()
                i += 1
                time.sleep(1 / self.fps)
            else:
                self.inputvideoCapture.release()
                break
    
    def segThread(self):
        # 语义分割线程
        logger.info("Thread Segmentation: start")

        # 处理输入视频帧（临时Fastdeploy对seg的支持问题，下个版本修复）
        frame = self.frame if self.device == 0 else cv2.resize(self.frame, (1024, 512))

        result, segImage = self.SegObject.predict(frame)
        segImage = cv2.cvtColor(segImage, cv2.COLOR_BGR2RGB)
        segImageQT = QImage(segImage, segImage.shape[1], segImage.shape[0], QImage.Format_RGB888)
        self.ui.segVideo.setPixmap(QPixmap.fromImage(segImageQT))
        self.ui.segVideo.setScaledContents(True)
        cv2.waitKey(1)
        self.ui.greenrate.setText("{} %".format(result["green"]))
        self.ui.buildingrate.setText("{} %".format(result["building"]))

        logger.info("Thread Segmentation: finished")

    def detThread(self):
        # 目标检测线程
        logger.info("Thread Detection: start")
        result, detImage = self.DetObject.predict(self.frame)
        # detImage = cv2.cvtColor(detImage, cv2.COLOR_BGR2RGB)
        detImageQT = QImage(detImage, detImage.shape[1], detImage.shape[0], QImage.Format_RGB888)
        self.ui.detVideo.setPixmap(QPixmap.fromImage(detImageQT))
        self.ui.detVideo.setScaledContents(True)
        cv2.waitKey(1)
        self.ui.humanCount.setText("{} 人".format(result["human"]))
        self.ui.carCount.setText("{} 辆".format(result["car"]))

        logger.info("Thread Detection: finished")


app = QApplication([])
window = App()
window.show()

sys.exit(app.exec())