# -- coding: utf-8 --
###svm进行手写体识别
import pandas as pd
from sklearn import  svm
import numpy as np
import datetime
start = datetime.datetime.now()
test_len = 5000
train_len = 25000
# 训练集，测试集读取
train = pd.read_csv('h_bit_all_train.csv',header=None)
test = pd.read_csv('h_bit_all_test.csv',header=None)
data_train = train.iloc[0:train_len, 1:]
label_train = train.iloc[0:train_len,0]

data_test = test.iloc[0:test_len, 1:]
label_test = test.iloc[0:test_len,0].T
label_test = np.array(label_test)

# 创建分类器
classifier = svm.SVC(C=200, kernel='rbf', gamma=0.01, cache_size=8000, probability=False)
classifier.fit(data_train, label_train)
#预测标签值
predicted = classifier.predict(data_test)
#获取平均准确率
index = np.arange(0,test_len)
T_len = index[label_test==predicted]
accuracy = float(len(T_len))/float(len(index))
print accuracy
###svm参数
print classifier.get_params()


df = pd.DataFrame(predicted)
df.index += 1
df.index.name = 'ImageId'
df.columns = ['Label']
df.to_csv('results.csv', header=True)
print(predicted)

end = datetime.datetime.now()
print (end-start).total_seconds()