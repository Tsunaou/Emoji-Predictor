

class Data_Option():


    def __init__(self) -> None:
        super().__init__()

        self.emojis_filename = 'EmojiDataSet/emoji.data'
        self.train_small_filename = 'EmojiDataSet/small_data/small_train.data'
        self.label_small_filename = 'EmojiDataSet/small_data/small_train.solution'
        self.train_filename = 'EmojiDataSet/train.data'
        self.test_filename = 'EmojiDataSet/test.data'
        self.label_filename = 'EmojiDataSet/train.solution'

        self.emojis_dict = {}  # 表情和数字的字典 {(No, Emoji)}
        self.train_dict = {}
        self.train_data = []
        self.train_label = []
        self.emojis_classfier = {}
        self.cxk_dict = {}
        self.stopwords = []
        self.test_data = []

    def get_test_data(self):
        """ 测试 """
        with open(self.test_filename, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                self.test_data.append(line.strip('\n').split('\t')[1])
            f.close()
        return self.test_data

    def get_stop_words(self):
        buff = []
        with open('files/stop', encoding='UTF-8-sig') as fp:
            for ln in fp:
                buff.append(ln.strip('\n'))
        self.stopwords = buff
        self.stopwords.append(' ')
        return self.stopwords

    def get_train_data(self,isSmall=False):
        if isSmall:
            self.train_filename = self.train_small_filename
            self.label_filename = self.label_small_filename

        """ 读取emoji对应关系 """
        with open(self.emojis_filename, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip('\n').split('\t')
                self.emojis_dict[tmp[1]] = tmp[0]
            f.close()

        """ 读取训练数据 """
        with open(self.train_filename, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                self.train_data.append(line.strip('\n'))
            f.close()

        """ 读取训练标签 """
        with open(self.label_filename, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                self.train_label.append(line.strip('\n').strip('{}'))
            f.close()

        assert (self.train_label.__len__() == self.train_data.__len__())

        for i in range(self.train_data.__len__()):
            '''这里就是用数字代替表情了'''
            self.train_dict[self.train_data[i]] = self.emojis_dict[self.train_label[i]]
            # self.train_dict[self.train_data[i]] = self.train_label[i]

        """ 根据表情对数据集分类 """
        for k, v in self.train_dict.items():

            """ 虚假的 """
            # if k.__contains__('cxk') or k.__contains__('蔡徐坤') or k.__contains__('kun'):
            #     # print(k,' --> ',v)
            #     if cxk_dict.get(v)==None:
            #         cxk_dict[v] = 1
            #     else:
            #         cxk_dict[v] = cxk_dict[v] + 1

            """ 真实的 """
            if self.emojis_classfier.get(v) == None:
                sentence_list = []
                sentence_list.append(k)
                self.emojis_classfier[v] = sentence_list
            else:
                self.emojis_classfier[v].append(k)

        # for k, v in self.emojis_classfier.items():
        #     if k != '心':
        #         continue
        #     print(k, ' --> ')
        #     for sentence in v:
        #         print('\t' + sentence)

        return self.emojis_dict,self.train_data,self.train_label,self.train_dict,self.emojis_classfier

