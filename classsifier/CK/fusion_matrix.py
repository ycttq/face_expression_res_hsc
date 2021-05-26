# -*- coding: utf-8 -*
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# classes表示不同类别的名称，比如这有6个类别
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise','Contempt']
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=8, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Classification label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()



y_true = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6] # 样本实际标签
 # 将前10个样本的值进行随机更改
y_pred = [0,0,0,0,0,6,2,0,2,2,4,0,4,0,0,2,0,0,0,2,2,6,0,0,0,0,0,1,1,1,1,1,1,1,1,5,1,2,0,0,2,2,2,2,0,2,2,2,2,2,2,0,2,2,2,2,0,0,0,2,5,2,2,2,0,2,2,2,4,4,0,2,2,0,2,3,0,6,3,4,3,3,4,3,3,1,3,3,4,4,4,4,4,4,4,4,4,2,2,2,3,4,2,2,4,4,0,2,6,4,0,4,4,6,4,4,4,0,4,4,4,4,4,4,4,3,2,4,4,5,5,0,2,0,5,0,5,5,5,4,5,0,3,3,1,5,4,6,6,6,2,4,6,6,6,6,6,4,6,4,6,6,6,0,6,6,6,3,6,2,2,5,6,6,6,6,6,6,4,6,6,6,6,4,4,5,1,6,3,6,6,5,5,6,6,6]  # 样本预测标签

# 获取混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')