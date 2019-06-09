import csv
import numpy as np
import pandas as pd


def my_csv_writer(pre,dsize):
    # 字典中的key值即为csv中列名
    a = []
    for i in range(len(pre)):
        a.append(i)
    b = []
    for i in range(len(pre)):
        b.append(int(pre[i]))
    dataframe = pd.DataFrame({'ID': a, 'Expected': b})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("EmojiDataSet/submission_"+str(dsize)+".csv", index=False, sep=',')

if __name__ == '__main__':

    # 任意的多组列表
    a = [1, 2, 3, 4, 5]
    b = np.zeros([5])

    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'ID': a, 'Expected': b})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("EmojiDataSet/submission.csv", index=False, sep=',')
