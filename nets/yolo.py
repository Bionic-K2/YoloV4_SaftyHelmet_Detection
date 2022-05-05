from collections import OrderedDict

import torch
import torch.nn as nn

from nets.CSPdarknet import darknet53


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53(pretrained)  #生成CSPdarknet53的主干模型

        self.conv1 = make_three_conv([512,1024],1024)  #三次卷积块
        self.SPP = SpatialPyramidPooling()  #利用不同大小的池化核进行池化,池化后堆叠
        self.conv2 = make_three_conv([512,1024],2048)  #三次卷积块

        self.conv_for_P4 = conv2d(512,256,1)  #卷积
        self.conv_for_P3 = conv2d(256,128,1) #卷积

        self.upsample1 = Upsample(512,256)  #卷积 + 上采样
        self.make_five_conv1 = make_five_conv([256, 512],512)  #五次卷积块

        self.upsample2 = Upsample(256,128)  #卷积 + 上采样
        self.make_five_conv2 = make_five_conv([128, 256],256)  #五次卷积块

        self.down_sample1       = conv2d(128,256,3,stride=2)  #下采样
        self.make_five_conv3    = make_five_conv([256, 512],512) #五次卷积块

        self.down_sample2       = conv2d(256,512,3,stride=2) #下采样
        self.make_five_conv4    = make_five_conv([512, 1024],1024) #五次卷积块

        self.yolo_head3 = yolo_head([256, num_anchors_mask * (5 + num_classes)],128) #YOLOV4输出
        self.yolo_head2 = yolo_head([512, num_anchors_mask * (5 + num_classes)],256)
        self.yolo_head1 = yolo_head([1024, num_anchors_mask * (5 + num_classes)],512)


    def forward(self, x):
        x2, x1, x0 = self.backbone(x) #获得三个有效特征层，对应框2

		#对第三特征层进行处理
        P5 = self.conv1(x0) #三次卷积
        P5 = self.SPP(P5) #池化
        P5 = self.conv2(P5) #三次卷积
        P5_upsample = self.upsample1(P5) #卷积+上采样，以上对应框3的第一条路线

		#对第二特征层进行处理
        P4 = self.conv_for_P4(x1) #卷积
        P4 = torch.cat([P4,P5_upsample],axis=1) #与第三层输出混合
        P4 = self.make_five_conv1(P4) #5次卷积
        P4_upsample = self.upsample2(P4) #卷积+上采样，以上对应框3的第二条路线

		#对第一特征层进行处理
        P3 = self.conv_for_P3(x2) #卷积
        P3 = torch.cat([P3,P4_upsample],axis=1) #与第二层输出混合
        P3 = self.make_five_conv2(P3) #5次卷积，以上对应框3的第1条输入线
        P3_downsample = self.down_sample1(P3)#卷积+下采样
        P4 = torch.cat([P3_downsample,P4],axis=1) #与p4混合
        P4 = self.make_five_conv3(P4) #5次卷积，以上对应框4的第2条输入线
        P4_downsample = self.down_sample2(P4) #卷积+下采样
        P5 = torch.cat([P4_downsample,P5],axis=1) #与p5混合
        P5 = self.make_five_conv4(P5) #5次卷积，以上对应框4的第3条输入线
        #   第三个特征层输出
        out2 = self.yolo_head3(P3)#卷积并获得yolov4的输出，框4的第1条输出
        #   第二个特征层输出
        out1 = self.yolo_head2(P4) #卷积并获得yolov4的输出，框4的第2条输出
        #   第一个特征层输出
        out0 = self.yolo_head1(P5) #卷积并获得yolov4的输出，框4的第3条输出
        return out0, out1, out2

