import math
import jieba
import os  # 用于处理文件路径
import re
import sys
import random
import numpy as np
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def read_novel(path_in, path_out):  # 读取语料内容
    content = []
    names = os.listdir(path_in)
    for name in names:
        novel_name = path_in + '/' + name
        fenci_name = path_out + '/' + name
        for line in open(novel_name, 'r'):
            line.strip('\n')
            line = re.sub("[A-Za-z0-9\：\·\—\，\。\“\”\\n \《\》\！\？\、\...]", "", line)
            line = content_deal(line)
            con = jieba.cut(line, cut_all=False) # 结巴分词
            content.append(" ".join(con))
        with open(fenci_name, "w", encoding='utf-8') as f:
            f.writelines(content)
    return names


def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = ['本书来自www.cr173.com免费txt小说下载站\n','更多更新免费电子书请关注www.cr173.com\n', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        content = content.replace(a, '')
    return content


if __name__ == '__main__':
    #dataset下为初始选定的五个语料文件，讲处理过的文件存储在output_dataset目录下
    files = read_novel("./dataset", "./output_dataset")
    test_name = ['黄蓉','韦小宝','小龙女','乔峰','赵敏']
    test_wugong = ['降龙十八掌','九阴真经','一阳指','降龙十八掌','亢龙有悔']
    for i in range(0, 5):
        name = "./output_dataset/" + files[i]
        model = Word2Vec(sentences=LineSentence(name), hs=1, min_count=10, window=5, vector_size=200, sg=0, epochs=200)
        for result in model.wv.similar_by_word(test_name[i], topn=10):
            print(result[0], result[1])
        for result in model.wv.similar_by_word(test_wugong[i], topn=10):
            print(result[0], result[1])
