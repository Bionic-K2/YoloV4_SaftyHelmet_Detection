import os
import sys
import csv  # csv输出模块
import time
import json
import codecs
import socket  # socket模块
import threading  # 多线程
from multiprocessing import Manager, Process, Queue
import cv2  # opencv
import numpy as np
import warnings  # 警告信息参数
import winsound

from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QTableWidgetItem,
)
from PyQt5.QtCore import Qt, QDir, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

from ui import UiMainWindow  # 界面

# -------------------------------------
# onnx文件通过 pytorch2onnx.py 转换生成
# -------------------------------------

# from detection.yolo_detection import *       #torch原始版本，最慢
# from detection.cv2_detection import *     #opencv+onnx版本,最快
# from detection.darknet_detection import *     #.weights+opencv版本
from detection.onnx_detection import *  # onnx版本


BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # 取程序运行文件夹
IMG_PATH = os.path.join(BASE_PATH, "img")  # 存放未戴安全帽视频截图
CFG_PATH = os.path.join(BASE_PATH, "model_data")  # 存放权重文件以及系统配置文件

fpbase = codecs.open(os.path.join(CFG_PATH, "config.ini"),
                     "r", "utf-8")  # 系统配置文件
str1 = json.dumps(eval(fpbase.read()))  # 读取文件内容，并转换为JSON字符串格式
cfg = json.loads(str1)  # 将JSON字符串反序列化为对象
fpbase.close()

# _weights：权重文件名
_weights = cfg["yolo_weights"] if "yolo_weights" in cfg.keys(
) else "ep100-loss1.539-val_loss1.490.pth"
# _config：权重配置文件，只有Darknet格式时有用
_config = cfg["yolo_config"] if "yolo_config" in cfg.keys(
) else "custom-yolov4-tiny-detector.cfg"

# 图片检测
# qs:图片输入队列
# qd:处理后的图片输出


def detect(qs, qd, index):
    # 检测代码，返回画框后的图片
    yolo = YOLODetect(CFG_PATH, _weights, _config, 416)  # 创建yolov4检测对象
    while True:  # 循环读取和处理
        frame = qs.get()  # 等待并读取队列数据
        # 检测方法，atype:报警类别，对应helmet_classes.txt, show_fpw:是否显示FPS,
        # confThreshold，nmsThreshold：阈值
        # 1表示person,也就是没戴安全帽的人，0为戴安全帽
        found, frame = yolo.get_processed_image(frame, atype=1, show_fps=False,
                                                confThreshold=0.5, nmsThreshold=0.3)
        # 将位置index,是否报警，输出图片组合在一起发送到队列
        qd.put((index, found, frame))

# 读取视频流
# qs:图片输出队列
# qd:消息输出队列
# monitor:监控位置对象


def get_image(qs, qd, monitor):
    capture = None

    while True:  # 循环
        inplay = monitor["inplay"]  # inplay:当前监控位置对象播放状态属性
        if inplay == "EXIT":  # 退出
            break  # 释放资源和关闭窗口
        if inplay == "STOP" and capture != None:  # 停止，则释放capture对象
            if capture.isOpened():
                capture.release()
            capture = None
        if inplay == "PLAY":  # 播放：打开视频流
            if capture == None and monitor["rtsp"] != "":  # 未打开
                if TestPort(monitor["ip"], monitor["port"]):  # 测试能否连接监控IP地址
                    try:
                        if(monitor["rtsp"].isdigit()):  # 如果地址为0，1，2等数字，说明是本地摄像头
                            capture = cv2.VideoCapture(int(monitor["rtsp"]))
                        else:
                            capture = cv2.VideoCapture(
                                monitor["rtsp"])  # 否则传入地址或视频文件名字符串
                        capture.set(cv2.CAP_PROP_FOURCC,
                                    cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
                        # 设置缓存为1张图片，保证获得的图像最新
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # -1:状态栏消息处理
                        qd.put((-1, 'msg', '成功连接到 '+monitor["name"]))
                    except Exception as e:
                        qd.put((-1, 'msg', '无法播放 '+monitor["name"]))
                        monitor["inplay"] = "STOP"  # 无法播放，更新状态
                else:
                    qd.put((-1, 'msg', '无法连接到RTSP地址:'+monitor["rtsp"]))
                    monitor["inplay"] = "STOP"
            if capture != None and capture.isOpened():
                try:
                    ret, frame = capture.read()  # 读取摄像头图像数据，帧数据格式为BGR
                    if ret:
                        qs.put((frame[:, :, ::-1]))  # 将读取的帧数据从BGR转为RGB后发送到队列
                        if qs.qsize() > 1:  # 在输出队列拥挤时，删除之前的帧数据
                            qs.get()
                    cv2.waitKey(25)  # 等待25ms后读下一帧数据，并给cpu处理其他事务的时间
                    continue
                except:
                    pass
        time.sleep(0.01)  # 在不读取视频流时，不能用cv2.waitkey

    if capture != None:
        capture.release()

# 检测端口号和ip地址能否连上


def TestPort(ip, port):
    if(ip == "" or port == 0):
        return True
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建一个socket对象
    sock.settimeout(5)  # 设置连接超时
    state = sock.connect_ex((ip, port))  # 开始连接
    if 0 == state:  # 如果返回0，说明能连上
        return True
    else:
        return False


def beep():
    winsound.Beep(4000, 100)

# 界面主程序


class MainPage(QMainWindow, UiMainWindow):

    # 创建pyqt信号，相当于c#或java中的事件。
    # python规定，在线程中不可以直接操作界面元素，必须通过信号（事件）方式触发
    playStatus = pyqtSignal(bool)  # 播放状态
    updateImage_signal = pyqtSignal(np.ndarray)  # 视频窗口更新
    updateStatus_signal = pyqtSignal(str)  # 状态栏更新
    processAlert_signal = pyqtSignal(int, np.ndarray)  # 报警信号处理

    def __init__(self, parent=None):
        super(MainPage, self).__init__(parent)
        # 调入参数
        fpbase = codecs.open(os.path.join(
            CFG_PATH, "config.ini"), "r", "utf-8")
        str1 = json.dumps(eval(fpbase.read()))
        cfg = json.loads(str1)  # 打开config.ini读取监控位置对象列表
        fpbase.close()

        self.monitors = []
        count = 0
        index = 0
        manager = Manager()
        while True:  # 从1开始查看是否有此键值
            words = cfg[str(count+1)] if str(count+1) in cfg.keys() else None
            if words != None:  # 找到了
                if (words["ip"] == "" and os.path.exists(words["rtsp"])) or (words["ip"] != "" and TestPort(words["ip"], words["port"])):
                    # 测试能否连接监控IP地址
                    _words = manager.dict()  # 需在多进程使用，所以必须用multiprocessing.Manager创建变量

                    _words["name"] = words["name"]  # 位置名称
                    _words["ip"] = words["ip"]  # ip地址
                    _words["port"] = words["port"]  # 端口号
                    _words["rtsp"] = words["rtsp"]  # rtsp地址，或文件名，或本地摄像头index
                    # 报警器WIFI继电器ip,端口号统一为8080
                    _words["alertIp"] = words["alertIp"]

                    _words["index"] = index  # index
                    _words["inplay"] = "STOP"  # 初始状态不播放
                    _words["frame"] = None  # 当前帧内容
                    _words["lastAlert"] = 0  # 最后一次报警时间
                    index += 1
                    self.monitors.append(_words)
                count += 1  # 加一
            else:
                break

        # 绘制界面
        self.setupUi(self)
        for idx, x in enumerate(self.monitors):
            self.rtsp_list.addItem(x["name"], idx)  # 更新界面上的监控位置列表

        # 事件关联
        self.test_time = QTimer()  # 右下角时间
        self.test_time.timeout.connect(self.time_convert)  # 每秒一次显示当前时间
        self.test_time.start(1000)

        if not os.path.exists(IMG_PATH):
            os.makedirs(IMG_PATH)  # 创建存放报警截图的文件夹

        self.rtsp_list.currentIndexChanged.connect(self.set_rtsp_url)  # 监控位置改变
        self.playButton.clicked.connect(self.play)  # 按下play按钮
        self.stopButton.clicked.connect(self.stop)  # stop按钮
        self.viewButton.clicked.connect(self.checkTableFrame)  # 查看按钮
        self.clearButton.clicked.connect(self.clearTable)  # 清空按钮
        self.exportButton.clicked.connect(self.export)  # 导出按钮
        self.exitButton.clicked.connect(self.clickExit)  # 退出按钮

        self.playStatus.connect(self.setbuttonstatus)  # 定义信号执行方法
        self.updateImage_signal.connect(self.updateImage)
        self.updateStatus_signal.connect(self.update_statusbar)
        self.processAlert_signal.connect(self.processAlert)

        self.currentmonitor = self.monitors[0]  # 初始化当前位置为第一个监控
        qd = Queue()  # 处理结果输出队列，同时用作消息队列（index<0时）

        t = threading.Thread(target=self.process_image,
                             args=(qd,))  # 更新视频框图像的线程
        t.setDaemon(True)  # 设为监护模式，主程序退出则线程自动退出
        t.start()

        self.processes = []
        queues = [Queue() for _ in self.monitors]  # 创建多进程用的待检测图片输出队列
        for qs, monitor in zip(queues, self.monitors):  # 每个监控对象分配一个独立的输出队列
            self.processes.append(
                Process(target=get_image, args=(qs, qd, monitor)))  # 读视频流
            self.processes.append(
                Process(target=detect, args=(qs, qd, monitor["index"])))  # 检测

        for process in self.processes:
            process.daemon = True
            process.start()

        self.rtsp_list.setCurrentIndex(0)  # 初始化监控位置列表index=0
        self.set_rtsp_url()

    # 显示状态栏的当前时间信息
    def time_convert(self):
        self.stb_time.setText(time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    # 显示状态栏的状态信息
    def update_statusbar(self, msg):
        self.stb_content.setText(msg)

    # 发送socket数据
    def sendSocket(self, ip, port, data):
        try:
            addr = (ip, port)
            __clientsocket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)  # 定义socket类型，网络通信，TCP
            __clientsocket.settimeout(5)
            __clientsocket.connect(addr)
        except Exception as e:
            return -1

        try:
            time.sleep(0.1)  # 等待初始化
            __clientsocket.sendall(data)  # 发送数据
            time.sleep(0.2)  # 等待继电器做出反应
            __clientsocket.close()  # 关闭socket
        except Exception as e:
            __clientsocket.close()
            return -1
        return 0

    # 播放
    def play(self):
        self.currentmonitor["inplay"] = "PLAY"
        self.setbuttonstatus()

    # 停止
    def stop(self):
        self.currentmonitor["inplay"] = "STOP"
        self.setbuttonstatus()

    # 更新播放停止按钮状态
    def setbuttonstatus(self):
        if(self.currentmonitor["inplay"] == "PLAY"):
            self.playButton.setEnabled(False)
            self.stopButton.setEnabled(True)
        else:
            self.playButton.setEnabled(True)
            self.stopButton.setEnabled(False)

    # 增加一条报警记录
    def addrecord(self, idx, time1, info, frame):
        timetxt = time.strftime("%y-%m-%d %H:%M:%S",
                                time.localtime(time1))  # 取报警时间
        position = self.monitors[idx]["name"]  # 取监控位置
        row = self.tableWidget.rowCount()
#        if(row>30):
#            self.clearTable()
#            row = 0
        self.tableWidget.setRowCount(row + 1)  # 在列表最后增加一行
        self.tableWidget.setItem(
            row, 0, QTableWidgetItem(timetxt))  # 设置 时间 位置 信息等属性
        self.tableWidget.setItem(row, 1, QTableWidgetItem(position))
        self.tableWidget.setItem(row, 2, QTableWidgetItem(info))
        timetxt = time.strftime("%y%m%d%H%M%S", time.localtime(time1))
        file = os.path.join(IMG_PATH, timetxt + str(idx) +
                            ".jpg")  # 定义报警截图文件名，以时间+站点号定义
        # 写入文件，cv2保存需按BGR格式
        cv2.imwrite(file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # 发送报警信号
    def alert(self, ip, data1, data2, time1):
        beep()
        self.sendSocket(ip, 8080, bytearray.fromhex(data1))  # 发送继电器闭合指令
        time.sleep(time1)  # 延时
        self.sendSocket(ip, 8080, bytearray.fromhex(data2))  # 发送继电器释放指令

    # 处理报警记录
    def processAlert(self, index, frame):
        # 异步操作如果不接收结果，系统会有警告，这里发完不管，加此代码禁止报警
        warnings.simplefilter('ignore', RuntimeWarning)
        time1 = time.time()
        monitor = self.monitors[index]  # 取监控位置对象
        lastAlert = monitor["lastAlert"]  # 取末次报警时间
        # 超过6秒，即使是同一报警也需要重复记录（一次语音报警播放完大约5秒）
        if time1 - lastAlert > 6:
            monitor["lastAlert"] = time1

        # 防误报，延续时长超过1秒，并且在2秒内重复，认为报警真实有效
        if(time1 - lastAlert > 1 and time1 - lastAlert < 2):
            monitor["lastAlert"] = lastAlert - 1  # 避免重复触发
            # 统一设置安全帽报警为第3路
#            self.alert(monitor["alertIp"],"A0 01 01 A2","A0 01 00 A1",1)   #第1路
#            self.alert(monitor["alertIp"],"A0 02 01 A3","A0 02 00 A2",1)   #第2路
#            self.alert(monitor["alertIp"],"A0 04 01 A5","A0 04 00 A4",1)   #第4路
            p = threading.Thread(target=self.alert, args=(
                monitor["alertIp"], "A0 03 01 A4", "A0 03 00 A3", 1))  # 第3路
            p.setDaemon(True)
            p.start()
            self.addrecord(index, time1, '未戴安全帽', frame)  # 增加记录

    # 处理检测后数据
    def process_image(self, qd):
        while True:
            index, found, frame = qd.get()
            if index < 0:  # 是消息队列
                if found == 'msg':  # 是状态信息，发送更新状态信号
                    self.updateStatus_signal.emit(frame)
                continue

            if (self.currentmonitor["index"] == index):  # 其他的信息，index表示图片来只哪个监控
                self.updateImage_signal.emit(frame)  # 如果是当前监控，发送更新图像信号
            if (found):  # 如果有报警，发送报警处理信号
                self.processAlert_signal.emit(index, frame)

    # 显示图像
    def updateImage(self, frame):
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, QImage.Format_RGB888)
        self.videoWidget.setPixmap(QPixmap.fromImage(img))

    # 清空列表
    def clearTable(self):
        while self.tableWidget.rowCount() > 0:
            self.tableWidget.removeRow(0)

    # 导出csv文件
    def export(self):
        # 显示保存为。。。对话框，并获取保存文件名
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save File', QDir.homePath() + "/report.csv", "CSV Files(*.csv *.txt)")
        if path:
            with open(path, 'w', newline='') as stream:  # 写方式打开
                writer = csv.writer(stream)
                model = self.tableWidget.model()  # 获取列表对象
                columnHeaders = []
                for column in range(model.columnCount()):  # 获取表头标题栏
                    item = model.headerData(column, Qt.Horizontal)
                    columnHeaders.append(item)
                writer.writerow(columnHeaders)  # 写表头标题栏

                for row in range(model.rowCount()):  # 遍历列表
                    rowdata = []
                    for column in range(model.columnCount()):  # 获取一行数据
                        item = model.data(model.index(row, column))
                        rowdata.append(item)
                    writer.writerow(rowdata)  # 写入一行数据

    # 点击查看显示当前时间点和位置的截图
    def checkTableFrame(self):
        index = self.tableWidget.currentIndex().row()  # 当前行

        if(index < 0):
            return
        position = self.tableWidget.item(index, 1).text()  # 位置栏内容
        for idx, x in enumerate(self.monitors):
            if x["name"] == position:  # 获取该位置名称对应的序号
                vtime = self.tableWidget.item(index, 0).text().replace(" ", "")
                vtime = vtime.replace("-", "")
                vtime = vtime.replace(":", "")
                file = os.path.join(
                    IMG_PATH, vtime + str(idx) + ".jpg")  # 还原文件名
                if os.path.isfile(file):
                    img = cv2.imread(file, 1)
                    cv2.imshow("Show Picture", img)  # 显示图片
                break

    # 退出前的清理动作
    def clickExit(self):
        for x in self.monitors:
            x["inplay"] = "EXIT"  # 向所有监控对象都发送退出信号
        time.sleep(3)  # 等待0。5秒让进程够时间结束
        for process in self.processes:
            process.terminate()
        time.sleep(0.2)  # 等待0。5秒让进程够时间结束
        sys.exit()

#    def closeEvent(self, event):
#        event.ignore()

    # 切换监控位置时，需要更新当前位置属性
    def set_rtsp_url(self, ):
        index = self.rtsp_list.currentData()
        self.currentmonitor = self.monitors[index]
        self.setbuttonstatus()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainPage()
    widget.show()
    app.exec_()
    widget.clickExit()
