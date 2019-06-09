from aip import AipNlp

""" 你的 APPID AK SK """
APP_ID = '16423779'
API_KEY = 'cyowbGYAc8Ftp1RBBDipsYBZ'
SECRET_KEY = 'nf8Dnl5Q0Iv6D1vWwPXQD8AH20uStAAE'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

text = "百度是一家辣鸡公司"

""" 调用词法分析 """
# 情感分析
sentimentAns = client.sentimentClassify(text)

print(sentimentAns)