import torch 
import numpy as np

'''用梯度下降来更新W——线性回归'''

# 给一组数据
x_data = np.array([1,2,3,4,5])
y_data = np.array([2,4,6,8,10])

w = torch.Tensor([1.0])
w.requires_grad=True

def forward(x):
	return x*w

def loss(x,y):
	y_pre = forward(x)
	return (y-y_pre)**2

print('训练之前预测x=6',forward(6).item())
for epoch0 in range(100):
	for x,y in zip(x_data,y_data):
		loss_val = loss(x,y)
		loss_val.backward()
		print('\tgrad:',x,y,w.grad.item())
		w.data = w.data-0.01*w.grad.data

		w.grad.data.zero_()
	print(f'epoch:{epoch0},cost:{loss_val},w:{w.data}')
print('训练之后预测x=6',forward(6).item())
