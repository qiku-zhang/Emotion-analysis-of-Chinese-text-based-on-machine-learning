import torch.nn as nn
import torch

class TextCNN(nn.Module):

    def __init__(self):
        super(TextCNN, self).__init__()
        output_channel = 5
        self.conv1 = nn.Sequential(nn.Conv2d(1, output_channel, (2,100)), # inpu_channel, output_channel, 卷积核高和宽 n-gram 和 embedding_size
                                nn.BatchNorm2d(output_channel),
                                nn.ReLU(),
                                nn.MaxPool2d((2,1)))                       
        self.fc = nn.Linear(70,2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def forward(self, X):
      '''
      X: [batch_size, sequence_length]
      '''
      batch_size = X.shape[0]
      embedding_X = X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
      conved = self.conv1(embedding_X) # [batch_size, output_channel,1,1]
      flatten = conved.view(batch_size, -1)# [batch_size, output_channel*1*1]
      output = self.fc(flatten)
      return output

class SpatialAttention(nn.Module): 
    def __init__(self, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False, act_func='relu', **kwargs):
        super().__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = nn.PReLU()
        self.relu = nn.ReLU()

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return self.relu(x + x * scale)


class AttTextCNN(nn.Module):

    def __init__(self):
        super(AttTextCNN, self).__init__()
        output_channel = 5
        self.att = SpatialAttention(kernel_size=(3,1),padding=1)
        self.conv1 = nn.Sequential(nn.Conv2d(1, output_channel, (2,100)), # inpu_channel, output_channel, 卷积核高和宽 n-gram 和 embedding_size
                                nn.BatchNorm2d(output_channel),
                                nn.ReLU(),
                                nn.MaxPool2d((2,1)))                       
        self.fc = nn.Linear(210,2)
        self.sf = nn.Softmax(dim=1)
    def forward(self, X):
      '''
      X: [batch_size, sequence_length]
      '''
      batch_size = X.shape[0]
      embedding_X = X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
      conved1 = self.conv1(embedding_X) # [batch_size, output_channel,14,1]
      att = self.att(conved1)
      flatten = att.view(batch_size, -1)# [batch_size, output_channel*14*1]
      output1 = self.fc(flatten)
      output = self.sf(output1)
      return output1