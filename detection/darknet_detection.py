import os
import time
import cv2
import torch

class YOLODetect(object):

    def __init__(self,base_path,weights,config,input_size=416 ):
        weights = 'custom-yolov4-tiny-detector.weights'  #测试用，正式应用请使用配置文件的配置，见主文件
        self.weightsPath = os.path.join(base_path,weights) #取得weights文件完整路径
        self.configPath = os.path.join(base_path,config) #取得cfg文件完整路径
        self.input_size = input_size
        self.COLORS = [(255,0,0),(0,255,0),(0,0,255)]   #按classes序号定义边框颜色
        self.cuda = (True if torch.cuda.is_available() else False)  #判断是否支持GPU
        
        self.net = self.create_detection_net()
        
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

    def create_detection_net(self,):
        net = cv2.dnn_DetectionModel(self.configPath, self.weightsPath)
        net.setInputSize(self.input_size, self.input_size)
        net.setInputScale(1.0 / 255)
        net.setInputSwapRB(True)
        if self.cuda:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def get_processed_image(self,image, atype, show_fps=False, confThreshold=0.5, nmsThreshold=0.4):
        atype=0     #演示weights文件中person为0
        found=False     #报警标志
        t1 = time.time()        
		#检测
        classes, confidences, boxes = self.net.detect(image, confThreshold, nmsThreshold)
        if len(classes)==0: #未检测到人和安全帽，直接返回原图
            return found, image
        if show_fps:
            fps  = 1./(time.time()-t1)
            image = cv2.putText(image, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for cl, score, (left, top, width, height) in zip(classes, confidences, boxes):
            if(cl==atype):
                found = True
            start_point = (int(left), int(top))
            end_point = (int(left + width), int(top + height))
            color = [int(c) for c in self.COLORS[cl]]
            image = cv2.rectangle(image, start_point, end_point, color, 2)  # draw class box

        return found,image


