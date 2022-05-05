import os
import time
import cv2
import numpy as np
import torch
from nets.yolo import YoloBody
from utils.utils_bbox import DecodeBox


class YOLODetect(object):

    def __init__(self, base_path, weights, config, input_size=416):
        self.weightsPath = os.path.join(base_path, weights)  # 取得pth文件完整路径
        self.input_shape = (input_size, input_size)  # 定义图片用于检测时resize尺寸
        self.num_classes = 2  # 两个分类，hat,person
        self.anchors = '12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401'
        self.anchors = [float(x) for x in self.anchors.split(',')]
        self.anchors = np.array(self.anchors).reshape(-1, 2)
        self.cuda = (True if torch.cuda.is_available() else False)  # 判断是否支持GPU

        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.COLORS = [(0, 255, 0), (255, 0, 0),
                       (0, 0, 255)]  # 按classes序号定义边框颜色
        self.bbox_util = DecodeBox(
            self.anchors, self.num_classes, self.input_shape, self.anchors_mask)  # 先验框
        self.create_detection_net()  # 创建检测网络

    def create_detection_net(self,):
        self.net = YoloBody(
            len(self.anchors_mask[0]), self.num_classes)  # 初始化网络模型
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # 根据是否支持GPU设置运算单元
        loads = torch.load(self.weightsPath, map_location=device)  # 调入权重文件
        self.net.load_state_dict(loads)
        del loads

        if self.cuda:  # 按是否支持GPU，确定运算代码
            self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.cuda()
        self.net.eval()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_processed_image(self, image, atype, show_fps=False, confThreshold=0.5, nmsThreshold=0.3):
        found = False  # 报警标志
        t1 = time.time()
        image_shape = np.array(np.shape(image)[0:2])  # 取得图片尺寸
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        images = np.array(cv2.resize(image, self.input_shape))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        images = np.expand_dims(np.transpose(
            (np.array(images, dtype='float32'))/255.0, (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
             # 将图像输入网络当中进行预测！
#            images = torch.autograd.Variable(images)
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # 将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, False, conf_thres=confThreshold, nms_thres=nmsThreshold)
            del outputs
            del images
            if results[0] is None:  # 未检测到人和安全帽，直接返回原图
                return found, image

        # 获得每个探测到物体最可信的classes_ID
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]  # 可信度
        top_boxes = results[0][:, :4]  # 边框位置

        if show_fps:
            fps = 1./(time.time()-t1)
            image = cv2.putText(image, "fps= %.2f" % (
                fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i, c in list(enumerate(top_label)):  # i:序号，c:classes_ID
            top, left, bottom, right = top_boxes[i]  # 获取边框的上下左右位置
            if(c == atype):  # 是否需报警物体
                found = True

            start_point = (int(left), int(top))  # 左上角
            end_point = (int(right), int(bottom))  # 右下角
            color = [int(c) for c in self.COLORS[c]]  # 对应颜色
            image = cv2.rectangle(image, start_point,
                                  end_point, color, 2)  # 用指定颜色画框
        return found, image
