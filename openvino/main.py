# -*- coding: utf-8 -*-

import cv2
import sys
from main_ui import Ui_MainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QImage, QPixmap
from threading import Thread
from OVInferEngine import SegModel, DetModel
import logging
import time
import numpy as np
from collections import Counter
from PIL import ImageDraw, Image

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
        self.threshold = float(self.ui.threshold.text()) # 检测置信度
        self.frameRate = int(self.ui.frame.text()) # 帧数

        # 初始化模型
        logger.info("Model Init.")

        self.SegObject = SegModel(num_threads=2, callback=self.segProcessCallback)
        self.DetObject = DetModel(num_threads=2, callback=self.detProcessCallback)

        self.inputvideoCapture = cv2.VideoCapture(self.inputVideoPath)
        self.fps = self.inputvideoCapture.get(cv2.CAP_PROP_FPS)
        self.display()

    def detProcessCallback(self, result, userdata):
        infer_result = result.output_tensors[0].data
        bbox_list = []
        label_dict = {0: 'animal', 1: 'rubbish', 2: 'bike', 3:'car', 4:'person'}
        label_num_dict = {'animal': 0, 'rubbish': 0, 'bike': 0, 'car': 0, 'person': 0}
        
        for r in range(len(infer_result)):
            cid, bbox, score = int(infer_result[r][0]), infer_result[r][2:], infer_result[r][1]

            for l in range(len(label_num_dict)):
                if cid == l and score >= self.threshold:
                    label_num_dict[label_dict[l]] += 1
                    bbox_list.append(bbox)

        self.ui.humanCount.setText("{} 人".format(label_num_dict['person']))
        self.ui.carCount.setText("{} 辆".format(label_num_dict['car']))
        self.ui.bikeCount.setText("{} 辆".format(label_num_dict['bike']))
        self.ui.animalCount.setText("{} 头".format(label_num_dict['animal']))
        self.ui.rubbishCount.setText("{} 处".format(label_num_dict['rubbish']))
        ori_img, scale_factor = userdata["ori_img"], userdata["scale_factor"]

        # 可视化
        detImage = self.drawBBox(ori_img, scale_factor, bbox_list)
        detImageQT = QImage(detImage, detImage.shape[1], detImage.shape[0], QImage.Format_RGB888)
        self.ui.detVideo.setPixmap(QPixmap.fromImage(detImageQT))
        self.ui.detVideo.setScaledContents(True)
        cv2.waitKey(1)


    def segProcessCallback(self, result, i):
        # 分割结果可视化
        infer_result = result.output_tensors[0].data
        result_vis_img = self.SegObject.get_pseudo_color_map(infer_result[0])
        segImage = cv2.cvtColor(np.array(result_vis_img), cv2.COLOR_RGB2BGR)
        segImageQT = QImage(segImage, segImage.shape[1], segImage.shape[0], QImage.Format_RGB888)
        self.ui.segVideo.setPixmap(QPixmap.fromImage(segImageQT))
        self.ui.segVideo.setScaledContents(True)
        cv2.waitKey(1)

        # 量化指标统计
        resultShape = infer_result[0].shape
        labelmapCount = dict(Counter(infer_result[0].flatten()))
        pixelTotal = int(resultShape[0] * resultShape[1])

        # 统计建筑率和绿地率
        buildingRate, greenRate = 0, 0
        if 8 in labelmapCount:
            buildingRate = round(labelmapCount[8] / pixelTotal* 100, 3) 
        if 9 in labelmapCount:
            greenRate = round(labelmapCount[9] / pixelTotal * 100 , 3)

        self.ui.greenrate.setText("{} %".format(greenRate))
        self.ui.buildingrate.setText("{} %".format(buildingRate))
        
    def display(self):
        i = 0
        while self.inputvideoCapture.isOpened():
            self.ret, self.frame = self.inputvideoCapture.read()
            if self.ret:
                self.frame_show = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = QImage(self.frame_show, self.frame_show.shape[1], self.frame_show.shape[0], QImage.Format_RGB888)
                self.ui.inputVideo.setPixmap(QPixmap.fromImage(image))
                self.ui.inputVideo.setScaledContents(True)
                cv2.waitKey(1)

                # 每N帧处理分割检测一次，防止使用GPU时显存不足
                if i % self.frameRate == 0:
                    self.SegObject.async_infer(self.frame)
                    self.DetObject.async_infer(self.frame)
                i += 1
                time.sleep(1 / self.fps)
            else:
                self.inputvideoCapture.release()
                
    def drawBBox(self, img, scale_factor, bbox_list):
        # 画框
        vis_img = Image.fromarray(img)
        if len(bbox_list) > 0:
            draw = ImageDraw.Draw(vis_img)
            y_scale, x_scale = scale_factor[0]
            for bbox in bbox_list:
                xmin, ymin, xmax, ymax = bbox
                xmin, ymin, xmax, ymax = int(xmin*x_scale), int(ymin*y_scale), int(xmax*x_scale), int(ymax*y_scale)
                draw.line(
                        [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                        (xmin, ymin)],
                        width=3,
                        fill="red")

        return cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)


app = QApplication([])
window = App()
window.show()

sys.exit(app.exec())