import numpy as np
from PIL import Image
from openvino.runtime import Core, AsyncInferQueue
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
from openvino.runtime import Layout, Type

class SegModel(object):
    def __init__(self, num_threads, callback) -> None:
        # 读取模型
        ie = Core()
        model = ie.read_model(r"./inference/ppliteseg/model.pdmodel")

        # 预处理
        ppp = PrePostProcessor(model)
        ppp.input().tensor() \
            .set_spatial_dynamic_shape() \
            .set_layout(Layout('NHWC')) \
            .set_color_format(ColorFormat.BGR)

        ppp.input().model().set_layout(Layout('NCHW'))

        ppp.input().preprocess() \
            .convert_color(ColorFormat.RGB) \
            .resize(ResizeAlgorithm.RESIZE_LINEAR,512,1024) \
            .mean([0.485, 0.456,0.406]) \
            .scale([0.229, 0.224, 0.225])

        ppp.output().tensor().set_element_type(Type.f32)

        model = ppp.build()

        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        self.compiled_model = ie.compile_model(model=model, device_name="CPU", config=config)
        
        # 设置多线程
        self.async_queue = AsyncInferQueue(self.compiled_model, num_threads)
        self.async_queue.set_callback(callback)

    def async_infer(self, img):
        img = np.expand_dims(img, 0) / 255
        result_infer = self.async_queue.start_async(inputs=img)

    def get_color_map_list(self, num_classes, custom_color=None):
        """
        Code from https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/paddleseg/utils/visualize.py

        Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
        Args:
            num_classes (int): Number of classes.
            custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.
        Returns:
            (list). The color map.
        """

        num_classes += 1
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = color_map[3:]

        if custom_color:
            color_map[:len(custom_color)] = custom_color
        return color_map

    def get_pseudo_color_map(self, pred, color_map=None):
        """
        Get the pseudo color image.
        Code from https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/paddleseg/utils/visualize.py
        Args:
            pred (numpy.ndarray): the origin predicted image.
            color_map (list, optional): the palette color map. Default: None,
                use paddleseg's default color map.
        Returns:
            (numpy.ndarray): the pseduo image.
        """
        pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')
        if color_map is None:
            color_map = self.get_color_map_list(256)
        pred_mask.putpalette(color_map)
        return pred_mask.convert('RGB')


class DetModel(object):
    def __init__(self, num_threads, callback) -> None:
        ie = Core()
        model = ie.read_model(r"./inference/ppyolo/model.pdmodel")

        # prepostcessing
        ppp = PrePostProcessor(model)

        ppp.input(1).tensor() \
            .set_spatial_dynamic_shape() \
            .set_layout(Layout('NHWC')) \
            .set_color_format(ColorFormat.BGR)
        
        ppp.input(1).model().set_layout(Layout('NCHW'))

        ppp.input(1).preprocess() \
            .convert_color(ColorFormat.RGB) \
            .resize(ResizeAlgorithm.RESIZE_LINEAR,320,320) \
            .mean([0.485, 0.456,0.406]) \
            .scale([0.229, 0.224, 0.225])

        model = ppp.build()

        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        compiled_model = ie.compile_model(model=model, device_name="CPU", config=config)
        
        self.async_queue = AsyncInferQueue(compiled_model, num_threads)
        self.async_queue.set_callback(callback)

    def async_infer(self, img):
        ori_img = img.copy()
        im_shape = np.array([img.shape[:2]]).astype(np.float32)
        scale_factor = self.generate_scale(img)
        img = np.expand_dims(img, 0) / 255
        result_infer = self.async_queue.start_async(inputs={"scale_factor":scale_factor, "image":img.astype(np.float32), "im_shape":im_shape}, userdata={"ori_img":ori_img, "scale_factor":scale_factor})
    
    def generate_scale(self, img):
        resize_h, resize_w = [320, 320]
        img_scale_y = resize_h / float(img.shape[0])
        img_scale_x = resize_w / (img.shape[1])
        return np.array([[img_scale_y, img_scale_x]]).astype(np.float32)