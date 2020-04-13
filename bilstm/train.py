from bilstm.preprocess import read_file
from bilstm.model import BiLSTM_CRF
import torch
from torch import nn
import torch.utils.data as Data
from torch import optim

EMBEDDING_DIM = 50
HIDDEN_DIM = 4
epochs = 1
START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"B": 0, "M": 1, "E": 2, "S": 3, START_TAG: 4, STOP_TAG: 5}


def get_tag_to_ix(tag_seq_list):
    tag_ix_seq_list = []
    for tag_seq in tag_seq_list:
        tag_ix_seq = []
        for tag in tag_seq:
            tag_ix_seq.append(tag_to_ix[tag])
        tag_ix_seq_list.append(tag_ix_seq)
    return tag_ix_seq_list


def prepare_squence(char_seq_list, to_ix):
    char_id_seq_list = []
    for char_seq in char_seq_list:

        idxs = [to_ix[w] for w in char_seq]
        char_id_seq_list.append(idxs)

    return char_id_seq_list


word_to_ix = {}


def get_word_to_id(char_list):
    for sentence in char_list:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

import random
def data_loader(X,Y,batch_size):
    data_set = zip(X,Y)
    data_set = list(data_set)
    print('len of data_set is {}'.format(len(data_set)))
    # data_set_index = list(range(len(data_set)))
    random.shuffle(data_set)
    data_set_len = len(data_set)
    # print(type(data_set))
    i=0
    while i< data_set_len:
        index = i + batch_size - 1
        if index <data_set_len:
            yield data_set[i:i+32]
        else:
            yield data_set[i:]
        i = i+32
        print('i :{}'.format(i))

# from tqdm import tqdm
import tqdm

if __name__ == '__main__':
    file_path = 'C:\\Users\\15708\\Desktop\\HMM_and_CRFS\\trainCorpus.txt_utf8'
    word_list, char_list, tag_list = read_file(file_path)
    print('len(word_list) is {}'.format(len(word_list)))
    print('len(char_list) is {}'.format(len(char_list)))
    print('len(tag_list) is {}'.format(len(tag_list)))

    get_word_to_id(char_list)
    char_ix_seq_list = prepare_squence(word_list,word_to_ix)
    tag_ix_seq_list = get_tag_to_ix(tag_list)
    # print(type(char_ix_seq_list))
    # print(type(tag_ix_seq_list))
    loader = data_loader(char_ix_seq_list, tag_ix_seq_list, 32)
    # print(loader.__len__())
    # for step, samples in enumerate(loader):
    #     for sample in samples:
    #         batch_x, batch_y = sample
    #         print("step:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print('Epoch: {}, training'.format(epoch))
        for step,samples in enumerate(loader):
            # print('Step : {} , training'.format(step))
            for i, sample in enumerate(samples):
                batch_x, batch_y = sample
                sentence_in = torch.tensor(batch_x, dtype=torch.long)
                targets = torch.tensor(batch_y, dtype=torch.long)
                model.zero_grad()
                loss = model.neg_log_likelihood(sentence_in, targets)
                if i ==30:
                    print('Step : {},loss:{:.6f}'.format(step, loss.data[0]))
                loss.backward()
                optimizer.step()

    # 保存
    torch.save(model,'cws_large.model')
    torch.save(model.state_dict(),'cws_large_all.model')
    # '''
