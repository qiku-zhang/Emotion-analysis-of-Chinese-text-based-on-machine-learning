import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        texts = []
        for i in range(len(labels)):
            # print(type(data[i]))    # <class 'PIL.Image.Image'>
            im_tensor = data[i].to(torch.device("cpu"))
            texts.append((im_tensor, labels[i]))
        self.texts = texts                         # DataLoader通过getitem读取图片数据
    def __getitem__(self, index):
        fn, label = self.texts[index]
        return fn, label
    def __len__(self):
        return len(self.texts)
