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

def cost(x,y):
	cost = 0
	for x_curent,y_curent in zip(x,y):
		cost+=(forward_preprogettion(x_curent)-y_curent)**2
	return cost

def Gradient(x,y):
	gradient = 0
	for x_curent,y_curent in zip(x,y):
		gradient+= -2*x_curent*(x_curent*w-y_curent)
	return gradient/x.size





print('训练之前预测x=6',forward_preprogettion(6))
for epoch0 in range(100):
	cost_val = cost(x_data,y_data)
	w+=Gradient(x_data,y_data)*0.01
	print(f'epoch:{epoch0},cost:{cost_val},w:{w}')
print('训练之后预测x=6',forward_preprogettion(6))
