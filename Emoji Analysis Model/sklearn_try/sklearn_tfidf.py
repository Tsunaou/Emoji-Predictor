from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus = ['aaa ccc aaa aaa',
          'aaa aaa',
          'aaa aaa aaa',
          'aaa aaa aaa aaa',
          'aaa bbb aaa bbb aaa',
          'ccc aaa aaa ccc aaa'
         ]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
print(word)

# 获取每个词在该行（文档）中出现的次数
counts =  X.toarray()
print (counts)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
#tfidf = transformer.fit_transform(counts) #与上一行的效果完全一样
#print(tfidf)
print(tfidf.toarray())