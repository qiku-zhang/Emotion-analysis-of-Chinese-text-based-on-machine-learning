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
 
def snownlp_split_sentence(input_file, output_file):
    # 把停用词做成列表
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
 
    fin = open(input_file, 'r', encoding='utf-8')  # 以读的方式打开文件
    fout = open(output_file, 'w', encoding='utf-8')  # 以写的方式打开文件
 
    for eachline in fin:
        line = eachline.strip() # 去除每行首尾可能出现的空格
        line1 = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)              # 用正则去除一些特殊符号
        wordlist = SnowNLP(line1)  # 用snownlp分词，对每行内容进行分词
        outstr = ''
        for word in wordlist.words:
            if word not in stopwords:      # 判断是否在停用词表中
                outstr += word
                outstr += ' '
        fout.write(outstr.strip() + '\n')  # 将分词好的结果写入到输出文件
    fin.close()
    fout.close()

def pkuseg_split_sentence(input_file, output_file):
    # 把停用词做成列表
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
 
    fin = open(input_file, 'r', encoding='utf-8')  # 以读的方式打开文件
    fout = open(output_file, 'w', encoding='utf-8')  # 以写的方式打开文件
    
    seg = pkuseg.pkuseg()
    for eachline in fin:
        line = eachline.strip() # 去除每行首尾可能出现的空格
        line1 = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)              # 用正则去除一些特殊符号
        wordlist = list(seg.cut(line1))  # 用北大分词，对每行内容进行分词
        outstr = ''
        for word in wordlist:
            if word not in stopwords:      # 判断是否在停用词表中
                outstr += word
                outstr += ' '
        fout.write(outstr.strip() + '\n')  # 将分词好的结果写入到输出文件
    fin.close()
    fout.close() 

def jieba_split_sentence(input_file, output_file):
    # 把停用词做成列表
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
 
    fin = open(input_file, 'r', encoding='utf-8')  # 以读的方式打开文件
    fout = open(output_file, 'w', encoding='utf-8')  # 以写的方式打开文件
    

    for eachline in fin:
        line = eachline.strip() # 去除每行首尾可能出现的空格
        line1 = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)              # 用正则去除一些特殊符号
        wordlist = list(jieba.cut(line1))  # 用结巴分词，对每行内容进行分词
        outstr = ''
        for word in wordlist:
            if word not in stopwords:      # 判断是否在停用词表中
                outstr += word
                outstr += ' '
        fout.write(outstr.strip() + '\n')  # 将分词好的结果写入到输出文件
    fin.close()
    fout.close() 

 
pkuseg_split_sentence('train_data.txt', 'pkuseg_cut_train_data.txt')
pkuseg_split_sentence('test_data.txt', 'pkuseg_cut_test_data.txt')





