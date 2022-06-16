
from PyQt5 import QtWidgets
from srs import Ui_Dialog
from srs1 import Ui_Dialog1
import jieba                                #导入jieba模块
import re
import torch
#jieba.load_userdict("mydict.txt")           #加载自定义词典
import torch
from torch.utils.data import DataLoader
import numpy as np
from gensim.models import Word2Vec
from solver.dataloader import MyDataset
from textcnn import TextCNN
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from predict import predict_jieba,predict_pkuseg,predict_snownlp,predict_jieba1,predict_pkuseg1,predict_snownlp1

class mywindow(QtWidgets.QWidget, Ui_Dialog):
    def  __init__ (self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.pushButton_start.clicked.connect(self.process)
    def process(self):
        a=self.comboBox_fenci.currentText()
        b=self.comboBox__cixiangliang.currentText()
        c=self.comboBox_net.currentText()
        print(a,b,c)
        self.child_window = secwindow(a,b,c)
        self.child_window.show()

      

class secwindow(QtWidgets.QWidget, Ui_Dialog1):
    def  __init__ (self,a,b,c):
        super(secwindow, self).__init__()    
        self.setupUi(self)
        self.a=a
        self.b=b
        self.c=c
        if self.a=="jieba分词":
            if self.b=="Word2Vec":
                if self.c=="TextCNN": 
                    self.textBrowser.setText('2345/3000')
                    self.textBrowser_2.setText('jieba + word2vec + TextCNN')
                elif self.c=="AttTextCNN":
                    self.textBrowser.setText('2496/3000')
                    self.textBrowser_2.setText('jieba + word2vec + AttTextCNN')
            elif self.b=="embedding":
                if self.c=="TextCNN": 
                    self.textBrowser.setText('2395/3000')
                    self.textBrowser_2.setText('jieba + embedding + TextCNN')
                elif self.c=="AttTextCNN":
                    self.textBrowser.setText('2502/3000')
                    self.textBrowser_2.setText('jieba + embedding + AttTextCNN')
        if self.a=="pkuseg分词":
            if self.b=="Word2Vec":
                if self.c=="TextCNN": 
                    self.textBrowser.setText('2302/3000')
                    self.textBrowser_2.setText('pkuseg + word2vec + TextCNN')
                elif self.c=="AttTextCNN":
                    self.textBrowser.setText('2423/3000')
                    self.textBrowser_2.setText('pkuseg + word2vec + AttTextCNN') 
            elif self.b=="embedding":
                if self.c=="TextCNN": 
                    self.textBrowser.setText('2314/3000')
                    self.textBrowser_2.setText('pkuseg + embedding + TextCNN')
                elif self.c=="AttTextCNN":
                    self.textBrowser.setText('2444/3000')
                    self.textBrowser_2.setText('pkuseg + embedding + AttTextCNN')                    
        if self.a=="snownlp分词":
            if self.b=="Word2Vec":
                if self.c=="TextCNN": 
                    self.textBrowser.setText('2299/3000')
                    self.textBrowser_2.setText('SnowNLP + word2vec + TextCNN') 
                elif self.c=="AttTextCNN":
                    self.textBrowser.setText('2404/3000')
                    self.textBrowser_2.setText('SnowNLP + word2vec + AttTextCNN')
            elif self.b=="embedding":
                if self.c=="TextCNN": 
                    self.textBrowser.setText('2393/3000')
                    self.textBrowser_2.setText('SnowNLP + embedding + TextCNN') 
                elif self.c=="AttTextCNN":
                    self.textBrowser.setText('2504/3000')
                    self.textBrowser_2.setText('SnowNLP + embedding + AttTextCNN')  
                             
        self.pushButton.clicked.connect(self.process)

    def process(self):
        xxx = self.lineEdit.text()
        if self.a=="jieba分词":
            if self.b=="Word2Vec":
                if self.c=="TextCNN":
                    model1 = Word2Vec.load('jieba+word2vec.model')
                    model = torch.load('jieba+word2vec+TextCNN1.pt')                           
                    result=predict_jieba(model,model1,xxx)
                elif self.c=="AttTextCNN":
                    model1 = Word2Vec.load('jieba+word2vec.model')
                    model = torch.load('jieba+word2vec+AttTextCNN1.pt')                           
                    result=predict_jieba(model,model1,xxx)
            elif self.b=="embedding":
                if self.c=="TextCNN":
                    model = torch.load('jieba+embedding+TextCNN1.pt')                           
                    result=predict_jieba1(model,xxx)
                elif self.c=="AttTextCNN":
                    model = torch.load('jieba+embedding+AttTextCNN1.pt')                           
                    result=predict_jieba1(model,xxx)                                        

        if self.a=="pkuseg分词":
            if self.b=="Word2Vec":
                if self.c=="TextCNN":
                    model1 = Word2Vec.load('pkuseg+word2vec.model')
                    model = torch.load('pkuseg+word2vec+TextCNN1.pt')         
                    result=predict_pkuseg(model,model1,xxx)
                elif self.c=="AttTextCNN":
                    model1 = Word2Vec.load('pkuseg+word2vec.model')
                    model = torch.load('pkuseg+word2vec+AttTextCNN1.pt')                           
                    result=predict_pkuseg(model,model1,xxx)
            elif self.b=="embedding":
                if self.c=="TextCNN":
                    model = torch.load('pkuseg+embedding+TextCNN1.pt')         
                    result=predict_pkuseg1(model,xxx)
                elif self.c=="AttTextCNN":
                    model = torch.load('pkuseg+embedding+AttTextCNN1.pt')                           
                    result=predict_pkuseg1(model,xxx)                    

        if self.a=="snownlp分词":
            if self.b=="Word2Vec":
                if self.c=="TextCNN":
                    model1 = Word2Vec.load('snownlp+word2vec.model')
                    model = torch.load('snownlp+word2vec+TextCNN1.pt')         
                    result=predict_snownlp(model,model1,xxx)
                elif self.c=="AttTextCNN":
                    model1 = Word2Vec.load('snownlp+word2vec.model')
                    model = torch.load('snownlp+word2vec+AttTextCNN1.pt')                           
                    result=predict_snownlp(model,model1,xxx)
            elif self.b=="embedding":
                if self.c=="TextCNN":
                    model = torch.load('snownlp+embedding+TextCNN1.pt')         
                    result=predict_snownlp1(model,xxx)
                elif self.c=="AttTextCNN":
                    model = torch.load('snownlp+embedding+AttTextCNN1.pt')                           
                    result=predict_snownlp1(model,xxx) 

        self.lineEdit_2.setText('%s'%result)

if __name__=="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    ui = mywindow()
    ui.show()
    sys.exit(app.exec_())
