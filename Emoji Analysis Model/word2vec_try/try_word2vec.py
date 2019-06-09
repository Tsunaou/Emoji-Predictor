from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import requests

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

#爬取alice.txt文档
sample = requests.get("http://www.gutenberg.org/files/11/11-0.txt")
s = sample.text

#将换行符变为空格
f = s.replace("\n", " ")

data = []   #存储的是单词表

#迭代文本中的每一句话
for i in sent_tokenize(f):
    temp = []

    #将句子进行分词
    for j in word_tokenize(i):
        temp.append(j.lower())  #存储的的时候全变为小写

    data.append(temp)

#创建CBOW模型
model1 = gensim.models.Word2Vec(data, min_count=1,size=200, window=7)   #忽略出现次数小于1的词，size是每个单词的词向量的维度

#输出结果
print("Cosine similarity between 'alice' " + "and 'wonderland' - CBOW : ", model1.similarity('alice', 'wonderland'))    #计算两个单词之间的余弦距离，返回相似度
print("Cosine similarity between 'alice' " + "and 'machines' - CBOW : ", model1.similarity('alice', 'machines'))

#创建Skip-Gram模型
model2 = gensim.models.Word2Vec(data, min_count=1, size=200, window=7, sg=1)    #sg=1说明使用的是Skip-gram模型

#输入结果
print("Cosine similarity between 'alice' " + "and 'wonderland' - Skip Gram : ", model2.similarity('alice', 'wonderland'))
print("Cosine similarity between 'alice' " + "and 'machines' - Skip Gram : ", model2.similarity('alice', 'machines'))