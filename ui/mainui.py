# -*- coding: utf-8 -*-

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QPushButton, QComboBox, QHBoxLayout, 
        QLabel, QSizePolicy, QStyle, QVBoxLayout,QWidget, QStatusBar, QTableWidget, 
        QVBoxLayout, QHBoxLayout, QHeaderView,QAbstractItemView)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(951, 664)
        MainWindow.setWindowTitle("厂区安全帽检测")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.tableWidget = QTableWidget()  #列表
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(['时间','位置','信息'])
        self.tableWidget.setColumnWidth(0,150)  #宽度：150，80，自动伸展
        self.tableWidget.setColumnWidth(1,80)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setAlternatingRowColors(True)  #交替色
        self.tableWidget.horizontalHeader().setStyleSheet("QHeaderView::section{background-color:rgb(240, 240, 240);};")  #标题浅灰色
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)  #单选
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  #整行选择
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  #不可修改
        
        self.videoWidget = QLabel()  #视频框
        self.videoWidget.resize(1280,720)
#        self.videoWidget.setSizePolicy(sizePolicy)
        
        self.errorLabel = QLabel("监控位置:")
        self.errorLabel.setFixedWidth(75)

        self.playButton = QPushButton() #play按钮
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setFixedWidth(40)

        self.stopButton = QPushButton()  #stop按钮
        self.stopButton.setEnabled(False)
        self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stopButton.setFixedWidth(40)
        
        self.rtsp_list = QComboBox(self)  #位置下拉列表
        
        self.viewButton = QPushButton("查看")  #按钮
        self.clearButton = QPushButton("清空")
        self.exportButton = QPushButton("导出")
        self.exitButton = QPushButton("退出")

        controlLayout = QHBoxLayout()  #水平格子布局，监控位置水平4个控件
        controlLayout.addWidget(self.errorLabel)
        controlLayout.addWidget(self.rtsp_list)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.stopButton)

        feats = QHBoxLayout()  #水平格子布局,右下角4个按钮
        feats.addWidget(self.viewButton)
        feats.addWidget(self.clearButton)
        feats.addWidget(self.exportButton)
        feats.addWidget(self.exitButton)

        layout = QVBoxLayout()  #垂直格子布局，视频窗+监控位置
        layout.addWidget(self.videoWidget, 3)
        layout.addLayout(controlLayout)

        layout2 = QVBoxLayout() #垂直格子布局，列表+右下角4个按钮
        layout2.addWidget(self.tableWidget)
        layout2.addLayout(feats, 2)

        # Main plotBox
        plotBox = QHBoxLayout() #水平布局，左侧视频框+右侧列表框
        plotBox.addLayout(layout, 5)
        plotBox.addLayout(layout2, 2)

        MainWindow.setCentralWidget(self.centralwidget)  #应用整个布局到mainwindow
        self.centralwidget.setLayout(plotBox)
        
        self.statusbar = QStatusBar(MainWindow) #状态栏
        self.stb_author = QLabel("作者： 软件工程-吴松霖")
        self.stb_author.setFixedWidth(180)
        self.stb_content = QLabel()
        self.stb_time = QLabel()
        self.stb_time.setFixedWidth(160)
        self.statusbar.addWidget(self.stb_author,1)  #将上述3个QLabel加到状态栏
        self.statusbar.addWidget(self.stb_content,2)
        self.statusbar.addWidget(self.stb_time,0)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)  #应用到窗口

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
