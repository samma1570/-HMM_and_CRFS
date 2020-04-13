from bilstm.preprocess import read_file, tag_to_ix
from bilstm.model import *
import torch
from torch import nn
import torch.utils.data as Data
from torch import optim
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
epochs=2


_,content,label=read_file('C:\\Users\\15708\\Desktop\\HMM_and_CRFS\\bilstm\\word.txt')

def train_data(content,label):
    train_data=[]
    for i in range(len(label)):
        train_data.append((content[i],label[i]))
    return train_data
data=train_data(content,label)

word_to_ix = {}
for sentence, tags in data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)


stri = "改善人民生活水平，建设社会主义政治经济。"
precheck_sent = prepare_sequence(stri, word_to_ix)
# print(precheck_sent)

net = torch.load('cws.model')
net.eval()
label = net(precheck_sent)[1]
# print(label)

cws = []

# # print(label)
for i in range(len(label)):
    cws.extend(stri[i])
    if label[i] == 2 or label == 3:
        cws.append('/')
# print(cws)
str = ''
for i in cws:
    str = str + i
print('==========Chinese Word Segmentation=========\n')
print('输入未分词语句：\n')
print(stri + '\n')
print('分词结果：\n')
print(str + '\n')
print('====================Done!===================\n')