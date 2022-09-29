import cv2
import numpy as np
import fastdeploy as fd
from PIL import Image
from collections import Counter

def FastdeployOption(device=0):
    option = fd.RuntimeOption()
    if device == 0:
        option.use_gpu()
    else:
        # 使用OpenVino推理
        option.use_openvino_backend()
        option.use_cpu()
    return option


class SegModel(object):
    def __init__(self, device=0) -> None:
        self.segModel = fd.vision.segmentation.ppseg.PaddleSegModel(
            model_file = 'inference/ppliteseg/model.pdmodel',
            params_file = 'inference/ppliteseg/model.pdiparams',
            config_file = 'inference/ppliteseg/deploy.yaml',
            runtime_option=FastdeployOption(device)
        )
    
    def predict(self, img):
        segResult = self.segModel.predict(img)
        result = self.postprocess(segResult)
        visImg = fd.vision.vis_segmentation(img, segResult)
        return result, visImg
    
    def postprocess(self, result):
        resultShape = result.shape
        labelmap = result.label_map
        labelmapCount = dict(Counter(labelmap))
        pixelTotal = int(resultShape[0] * resultShape[1])
        # 统计建筑率和绿地率
        buildingRate, greenRate = 0, 0
        if 8 in labelmapCount:
            buildingRate = round(labelmapCount[8] / pixelTotal* 100, 3) 
        if 9 in labelmapCount:
            greenRate = round(labelmapCount[9] / pixelTotal * 100 , 3)
        
        return {"building": buildingRate, "green": greenRate}


class DetModel(object):
    def __init__(self, device=0) -> None:
        self.detModel = fd.vision.detection.PPYOLO(
            model_file = 'inference/ppyolo/model.pdmodel',
            params_file = 'inference/ppyolo/model.pdiparams',
            config_file = 'inference/ppyolo/infer_cfg.yml',
            runtime_option=FastdeployOption(device)
        )
        # 阈值设置
        self.threshold = 0.3
    
    def predict(self, img):
        detResult = self.detModel.predict(img.copy())
        result = self.postprocess(detResult)
        visImg = fd.vision.vis_detection(img, detResult, self.threshold, 2)
        return result, visImg

    def postprocess(self, result):
        # 得到结果
        detIds = result.label_ids
        detScores = result.scores
        # 统计数量
        humanNum, CarNum = 0, 0
        for i in range(len(detIds)):
            if detIds[i] == 0 and detScores[i] >= self.threshold:
                humanNum += 1
            if detIds[i] == 2 and detScores[i] >= self.threshold:
                CarNum += 1
        return {"human": humanNum, "car": CarNum}


        

