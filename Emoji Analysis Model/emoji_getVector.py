import json
from gensim.models import word2vec
import time

vec_nsize = 12

if __name__ == '__main__':

    start = time.process_time()
    """ 加载分词后的句子 """
    data_words1 = []
    data_words2 = []

    with open('files/data_words.txt','r', encoding='UTF-8-sig') as FR:
        data_words1 = json.loads(FR.read())

    FR.close()

    # with open('files/test_data_words.txt','r', encoding='UTF-8-sig') as FT:
    #     data_words2 = json.loads(FT.read())
    #
    # FT.close()

    data_words = data_words1 + data_words2
    """ 训练词向量模型 """
    model=word2vec.Word2Vec(data_words, size=vec_nsize,min_count=3)
    end = time.process_time()
    print('Running time: %s Seconds'%(end-start))

    """ 保存（加载）模型 """
    model.save('models/second_model_'+str(vec_nsize)+'d')

    end = time.process_time()
    print('Running time: %s Seconds'%(end-start))
