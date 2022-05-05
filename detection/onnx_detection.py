import os
import time
import cv2
import numpy as np
import onnxruntime
import torch
from utils.utils_bbox import *

class YOLODetect(object):

    def __init__(self,base_path,weights,config,input_size=416 ):
        weights = 'yolov4_test.onnx'    #测试用，正式应用请使用配置文件的配置，见主文件
        self.weightsPath = os.path.join(base_path,weights)  #取得onnx文件完整路径
        self.configPath = os.path.join(base_path,config)    #onnx格式用不到
        self.input_shape = (input_size,input_size)  #定义图片用于检测时resize尺寸
        self.num_classes = 2    #两个分类，hat,person
        self.anchors = '12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401'
        self.anchors = [float(x) for x in self.anchors.split(',')]
        self.anchors = np.array(self.anchors).reshape(-1, 2)
        self.cuda = (True if torch.cuda.is_available() else False)  #判断是否支持GPU

        self.COLORS = [(0,255,0),(255,0,0),(0,0,255)]   #按classes序号定义边框颜色
        
        #创建检测网络，providers=['CPUExecutionProvider'])    #'CUDAExecutionProvider'：选cpu还是gpu
        provider = ('CUDAExecutionProvider' if self.cuda else 'CPUExecutionProvider')
        self.session = onnxruntime.InferenceSession(self.weightsPath,providers=[provider]) 
        self.input_name = self.session.get_inputs()[0].name
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, self.input_shape) #先验框

    '''
    以往利用dnn模块进行深度学习时

    通过net=cv2.dnn.readNetFromCaffe('*','*')加载模型，
    利用blob=cv2.dnn.blobFromImage进行图像预处理，
    利用 net.setInput(blob) detections = net.forward() 获取检测结果，
    最后对detections进行后处理，得到bounding box等相关信息

    '''
    '''
        if self.args.use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    '''

    def get_processed_image(self,image, atype, show_fps=False, confThreshold=0.5, nmsThreshold=0.4):
        found=False     #报警标志
        t1 = time.time()
        image_shape = np.array(np.shape(image)[0:2])  #取得图片尺寸
        #调整图片尺寸，并归一化为小数
        images = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_LINEAR)
        images = np.transpose(images, (2, 0, 1)).astype(np.float32)
        images = np.expand_dims(images, axis=0)
        images /= 255.0
        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
#            images = torch.autograd.Variable(images)
        outputs = self.session.run([], {self.input_name: images})
        outputs = self.bbox_util.decode_box(outputs) #取得预测框
        #---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        #---------------------------------------------------------#
        results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                    image_shape, False, conf_thres = confThreshold, nms_thres = nmsThreshold)
        del outputs
        del images
        if results[0] is None: #未检测到人和安全帽，直接返回原图
            return found, image

        top_label   = np.array(results[0][:, 6], dtype = 'int32') #获得每个探测到物体最可信的classes_ID
        top_conf    = results[0][:, 4] * results[0][:, 5]  #可信度
        top_boxes   = results[0][:, :4]  #边框位置

        if show_fps:
            fps  = 1./(time.time()-t1)
            image = cv2.putText(image, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i, c in list(enumerate(top_label)):  #i:序号，c:classes_ID
            top, left, bottom, right  = top_boxes[i]   #获取边框的上下左右位置
            if(c==atype):  #是否需报警物体
                found = True

            start_point = (int(left), int(top))  #左上角
            end_point = (int(right), int(bottom))  #右下角
            color = [int(c) for c in self.COLORS[c]]  #对应颜色
            image = cv2.rectangle(image, start_point, end_point, color, 2)  # 用指定颜色画框
        return found,image
