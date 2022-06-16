#jieba.load_userdict("mydict.txt")           #加载自定义词典
import torch
from torch.utils.data import DataLoader
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from solver.dataloader import MyDataset
from textcnn import TextCNN,AttTextCNN
import torch.nn as nn
import torch.optim as optim


#模型生成
model1 = Word2Vec(LineSentence('cut_train_data.txt'),vector_size=100, min_count=10,sg=1)
#模型保存
model1.save('snownlp+word2vec.model')
model1.wv.save_word2vec_format('word2vec.txt',binary=False) 
#循环遍历生成序列
model1 = gensim.models.Word2Vec.load('snownlp+word2vec.model')
 
dic = model1.wv.index_to_key
print(dic)
print(len(dic))

#训练集数据
data = [line.strip() for line in open('cut_train_data.txt', 'r', encoding='utf-8').readlines()] 
for i,line in enumerate(data):
    words = np.array(line.split())
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
    if i==0:
        data_raw = torch.unsqueeze(raw_datas,dim=0)
    else:
        data_raw = torch.cat((data_raw,torch.unsqueeze(raw_datas,dim=0)),dim=0)    

labels = [line.strip() for line in open('train_label.txt', 'r', encoding='utf-8').readlines()]
for i,line in enumerate(labels):
    if i==0:
        label_raw = torch.unsqueeze(torch.tensor(int(line)),dim=0)
    else:
        label_raw = torch.cat((label_raw,torch.unsqueeze(torch.tensor(int(line)),dim=0)),dim=0)
data_train=data_raw
label_train=label_raw 
print(data_train.shape)
print(label_train.shape)
train_data = MyDataset(data_train, label_train)
train_loader = DataLoader(dataset=train_data, batch_size=200, shuffle=True)


#测试集数据
data = [line.strip() for line in open('cut_test_data.txt', 'r', encoding='utf-8').readlines()] 
for i,line in enumerate(data):
    words = np.array(line.split())
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
    if i==0:
        data_raw = torch.unsqueeze(raw_datas,dim=0)
    else:
        data_raw = torch.cat((data_raw,torch.unsqueeze(raw_datas,dim=0)),dim=0)    

labels = [line.strip() for line in open('test_label.txt', 'r', encoding='utf-8').readlines()]
for i,line in enumerate(labels):
    if i==0:
        label_raw = torch.unsqueeze(torch.tensor(int(line)),dim=0)
    else:
        label_raw = torch.cat((label_raw,torch.unsqueeze(torch.tensor(int(line)),dim=0)),dim=0)
data_test=data_raw
label_test=label_raw 
print(data_test.shape)
print(label_test.shape)
test_data = MyDataset(data_test, label_test)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

model = AttTextCNN().to('cpu')
criterion = nn.CrossEntropyLoss().to('cpu')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(500):
  for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to('cpu'), batch_y.to('cpu')
    pred = model(batch_x)
    loss = criterion(pred, batch_y)
    if (epoch + 1) % 50 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# test_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)
torch.save(model, 'snownlp+word2vec+AttTextCNN1.pt')

right_count = 0
wrong_count = 0
for batch_x, batch_y in test_loader:
    model = model.eval()
    predict = model(batch_x).data.max(1, keepdim=True)[1]
    print(model(batch_x).data)
    print(model(batch_x).data.max(1, keepdim=True))
    if predict[0][0] == batch_y:
        right_count += 1
    else:
        wrong_count += 1

print('测试集中预测正确的数量为 %d/%d'%(right_count,wrong_count+right_count))