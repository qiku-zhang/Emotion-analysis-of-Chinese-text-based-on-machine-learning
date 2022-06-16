#-*- encoding:utf-8 -*-
import pkuseg 
import thulac
from snownlp import SnowNLP
import jieba                                #导入jieba模块
import re
#jieba.load_userdict("mydict.txt")           #加载自定义词典
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def predict_jieba(model,model1,xxx):
    line = xxx.strip() # 去除每行首尾可能出现的空格
    line1 = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)              # 用正则去除一些特殊符号
    wordlist = list(jieba.cut(line1))  # 用结巴分词，对每行内容进行分词
    outstr = ''
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
    for word in wordlist:
        if word not in stopwords:      # 判断是否在停用词表中
            outstr += word
            outstr += ' '

    data = outstr
    words = np.array(data.split())
    for index,word in enumerate(words):
        try:
            datas = torch.unsqueeze(torch.tensor(model1.wv['%s'%word]),dim=0)
        except:
            datas = torch.zeros([1,100],dtype=torch.float)
        if index==0:
            raw_datas = datas
        else:
            raw_datas = torch.cat((raw_datas,datas),dim=0)
        if index>=29:
            break 
    if raw_datas.shape[0]<30:
        raw_datas = torch.cat((raw_datas,torch.zeros([30-raw_datas.shape[0],100],dtype=torch.float)))
    raw_datas = torch.unsqueeze(raw_datas,dim=0)
    model_pre = model.eval()
    predict = model_pre(raw_datas).data.max(1, keepdim=True)[1]
    if predict[0][0] == 1:
        result='Positive'
    else:
        result='Negtive'
    return result

def predict_pkuseg(model,model1,xxx):
    seg = pkuseg.pkuseg()
    line = xxx.strip() # 去除每行首尾可能出现的空格
    line1 = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)              # 用正则去除一些特殊符号
    wordlist = list(seg.cut(line1))  # 用结巴分词，对每行内容进行分词
    outstr = ''
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
    for word in wordlist:
        if word not in stopwords:      # 判断是否在停用词表中
            outstr += word
            outstr += ' '

    data = outstr
    words = np.array(data.split())
    for index,word in enumerate(words):
        try:
            datas = torch.unsqueeze(torch.tensor(model1.wv['%s'%word]),dim=0)
        except:
            datas = torch.zeros([1,100],dtype=torch.float)
        if index==0:
            raw_datas = datas
        else:
            raw_datas = torch.cat((raw_datas,datas),dim=0)
        if index>=29:
            break 
    if raw_datas.shape[0]<30:
        raw_datas = torch.cat((raw_datas,torch.zeros([30-raw_datas.shape[0],100],dtype=torch.float)))
    raw_datas = torch.unsqueeze(raw_datas,dim=0)
    model_pre = model.eval()
    predict = model_pre(raw_datas).data.max(1, keepdim=True)[1]
    if predict[0][0] == 1:
        result='Positive'
    else:
        result='Negtive'
    return result

def predict_snownlp(model,model1,xxx):
    line = xxx.strip() # 去除每行首尾可能出现的空格
    line1 = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)              # 用正则去除一些特殊符号
    wordlist = SnowNLP(line1)  # 用结巴分词，对每行内容进行分词
    outstr = ''
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
    for word in wordlist.words:
        if word not in stopwords:      # 判断是否在停用词表中
            outstr += word
            outstr += ' '

    data = outstr
    words = np.array(data.split())
    for index,word in enumerate(words):
        try:
            datas = torch.unsqueeze(torch.tensor(model1.wv['%s'%word]),dim=0)
        except:
            datas = torch.zeros([1,100],dtype=torch.float)
        if index==0:
            raw_datas = datas
        else:
            raw_datas = torch.cat((raw_datas,datas),dim=0)
        if index>=29:
            break 
    if raw_datas.shape[0]<30:
        raw_datas = torch.cat((raw_datas,torch.zeros([30-raw_datas.shape[0],100],dtype=torch.float)))
    raw_datas = torch.unsqueeze(raw_datas,dim=0)
    model_pre = model.eval()
    predict = model_pre(raw_datas).data.max(1, keepdim=True)[1]
    if predict[0][0] == 1:
        result='Positive'
    else:
        result='Negtive'
    return result
# print(predict_jieba(model,model1,xxx))

def predict_snownlp1(model,xxx):
    temp = ""
    counts = {}
    data = [line.strip() for line in open('snownlp_cut_train_data.txt', 'r', encoding='utf-8').readlines()] 
    for line in data:
        words = np.array(line.split())
        for word in words:
            temp += word
            temp += '\n'
            counts[word] = counts.get(word, 0) + 1#统计每个词出现的次数

    dict = counts
    result=[]
    a = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    for i in range(1,101):
        result.append(a[i-1][0])
    
    line = xxx.strip() # 去除每行首尾可能出现的空格
    line1 = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)              # 用正则去除一些特殊符号
    wordlist = SnowNLP(line1)  # 用结巴分词，对每行内容进行分词
    outstr = ''
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
    for word in wordlist.words:
        if word not in stopwords:      # 判断是否在停用词表中
            outstr += word
            outstr += ' '

    data = outstr
    words = np.array(data.split())
    for index,word in enumerate(words):
        try:
            datas = torch.unsqueeze(torch.tensor(result.index(word)),dim=0)
        except:
            datas = torch.tensor([0])
        if index==0:
            raw_datas = datas
        else:
            raw_datas = torch.cat((raw_datas,datas),dim=0)
        if index>=29:
            break   
    if raw_datas.shape[0]<30:
        raw_datas = torch.cat((raw_datas,torch.zeros(30-raw_datas.shape[0],dtype=torch.int)))
    raw_datas = torch.unsqueeze(raw_datas,dim=0)
    model_pre = model.eval()
    predict = model_pre(raw_datas).data.max(1, keepdim=True)[1]
    if predict[0][0] == 1:
        result='Positive'
    else:
        result='Negtive'
    return result

def predict_jieba1(model,xxx):
    temp = ""
    counts = {}
    data = [line.strip() for line in open('jieba_cut_train_data.txt', 'r', encoding='utf-8').readlines()] 
    for line in data:
        words = np.array(line.split())
        for word in words:
            temp += word
            temp += '\n'
            counts[word] = counts.get(word, 0) + 1#统计每个词出现的次数

    dict = counts
    result=[]
    a = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    for i in range(1,101):
        result.append(a[i-1][0])
    
    line = xxx.strip() # 去除每行首尾可能出现的空格
    line1 = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)              # 用正则去除一些特殊符号
    wordlist = list(jieba.cut(line1))  # 用结巴分词，对每行内容进行分词
    outstr = ''
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
    for word in wordlist:
        if word not in stopwords:      # 判断是否在停用词表中
            outstr += word
            outstr += ' '

    data = outstr
    words = np.array(data.split())
    for index,word in enumerate(words):
        try:
            datas = torch.unsqueeze(torch.tensor(result.index(word)),dim=0)
        except:
            datas = torch.tensor([0])
        if index==0:
            raw_datas = datas
        else:
            raw_datas = torch.cat((raw_datas,datas),dim=0)
        if index>=29:
            break   
    if raw_datas.shape[0]<30:
        raw_datas = torch.cat((raw_datas,torch.zeros(30-raw_datas.shape[0],dtype=torch.int)))
    raw_datas = torch.unsqueeze(raw_datas,dim=0)
    model_pre = model.eval()
    predict = model_pre(raw_datas).data.max(1, keepdim=True)[1]
    if predict[0][0] == 1:
        result='Positive'
    else:
        result='Negtive'
    return result

def predict_pkuseg1(model,xxx):
    temp = ""
    counts = {}
    data = [line.strip() for line in open('pkuseg_cut_train_data.txt', 'r', encoding='utf-8').readlines()] 
    for line in data:
        words = np.array(line.split())
        for word in words:
            temp += word
            temp += '\n'
            counts[word] = counts.get(word, 0) + 1#统计每个词出现的次数

    dict = counts
    result=[]
    a = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    for i in range(1,101):
        result.append(a[i-1][0])
    
    seg=pkuseg.pkuseg()
    line = xxx.strip() # 去除每行首尾可能出现的空格
    line1 = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)              # 用正则去除一些特殊符号
    wordlist = list(seg.cut(line1))  # 用结巴分词，对每行内容进行分词
    outstr = ''
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
    for word in wordlist:
        if word not in stopwords:      # 判断是否在停用词表中
            outstr += word
            outstr += ' '

    data = outstr
    words = np.array(data.split())
    for index,word in enumerate(words):
        try:
            datas = torch.unsqueeze(torch.tensor(result.index(word)),dim=0)
        except:
            datas = torch.tensor([0])
        if index==0:
            raw_datas = datas
        else:
            raw_datas = torch.cat((raw_datas,datas),dim=0)
        if index>=29:
            break   
    if raw_datas.shape[0]<30:
        raw_datas = torch.cat((raw_datas,torch.zeros(30-raw_datas.shape[0],dtype=torch.int)))
    raw_datas = torch.unsqueeze(raw_datas,dim=0)
    model_pre = model.eval()
    predict = model_pre(raw_datas).data.max(1, keepdim=True)[1]
    if predict[0][0] == 1:
        result='Positive'
    else:
        result='Negtive'
    return result
