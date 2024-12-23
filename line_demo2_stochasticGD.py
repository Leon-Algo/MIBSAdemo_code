import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl 


'''用梯度下降来更新W——线性回归'''

# 给一组数据
x_data = np.array([1,2,3,4,5])
y_data = np.array([2,4,6,8,10])

#初始化权重
w = np.random.rand(x_data.ndim)


def forward_preprogettion(x):
	y_pre0 = x*w
	return y_pre0

def loss(x,y):
	loss = 0
	loss+=(forward_preprogettion(x)-y)**2
	return loss

def Gradient(x,y):
	gradient = 0
	gradient+= -2*x*(x*w-y)
	return gradient




print('训练之前预测x=6',forward_preprogettion(6))
for epoch0 in range(100):
	for x,y in zip(x_data,y_data):
		loss_val = loss(x,y)
		w+=Gradient(x,y)*0.01
	print(f'epoch:{epoch0},cost:{loss_val},w:{w}')
print('训练之后预测x=6',forward_preprogettion(6))
