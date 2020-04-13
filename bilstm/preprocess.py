import re
import torch
START_TAG = '<START>'
STOP_TAG = '<STOP>'

tag_to_ix = {"B": 0, "M": 1, "E": 2,"S":3, START_TAG: 4, STOP_TAG: 5}


def prepare_sequence(seq, to_ix): #seq是字序列，idxs是字序列对应的向量，to_ix是字和序号的字典
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def get_char_list(sentence):
    # word_list = []
    sentence = sentence.replace(' ','')
    return list(sentence)

def get_tap_list(sentence):
    tag_list =[]
    word_list = sentence.split(' ')
    for word in word_list:
        if len(word) == 1:
            tag_list.append('S')
        elif len(word) ==2:
            tag_list.append('B')
            tag_list.append('E')
        else:
            tag_list.append('B')
            tag_list.extend('M'*(len(word)-2))
            tag_list.append('E')
    return tag_list


def read_file(file_path):
    word_list=[]
    char_list=[]
    tag_list=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line_split = line.split(' ')
            char_list_1 = get_char_list(line)
            tag_list_1 = get_tap_list(line)
            word_list.extend(line_split)
            char_list.append(char_list_1)
            tag_list.append(tag_list_1)


    return word_list, char_list, tag_list







if __name__ == '__main__':
    sentence = '１９８６年 双方 协定 贸易额 达 二十六 亿 美元 ， '
    print(get_char_list(sentence))
    print(get_tap_list(sentence))