# -- coding: utf-8 --
from matplotlib import pyplot as plt
x = [1,4,16,64]#点的横坐标
res_ = [0.6051,0.641,0.653,0.7253]
plt.plot(x,res_,'s-',color='b')
plt.xlabel("Number of subintervals")#横坐标名字
plt.ylabel("accuracy")#纵坐标名字
plt.title('Multi-distribution feature statistical histogram')
plt.show()