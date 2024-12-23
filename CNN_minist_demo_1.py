import numpy as np
import torch
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset     #需要用DataLoader来划分batch数据，以及划分train和test数据集
import torch.nn.functional as F   #需要用到其中的RELU激活函数
# from sklearn.model_selection import train_test_split

'''
1 prepare training dataset
2 design model using Class inherit from nn.Module
3 construct loss function and optimizer
4 training cycle(forward->(zero_grad)backward->step(update))
5 visualize
'''

#1 prepare training dataset
batch_size = 64
# 对图像数据转为tensor以及进行标准化操作
transform_operation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
#transforms的一个集成操作（含totensor和normalize两个组合操作）

train_dataset = datasets.MNIST(root='./dataset/mnist/',train=True,download=False,transform=transform_operation)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

test_dataset = datasets.MNIST(root='./dataset/mnist/',train=False,download=False,transform=transform_operation)
test_loader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size)

#2 design model using Class inherit from nn.Module
class CNurelNet(torch.nn.Module):
	def __init__(self):
		super(CNurelNet,self).__init__()
		self.converlusion_1 = torch.nn.Conv2d(1,10,kernel_size = 5)
		self.converlusion_2 = torch.nn.Conv2d(10,20,kernel_size = 5)
		self.pooling = torch.nn.MaxPool2d(2)
		self.fullconnect = torch.nn.Linear(320,10)

	def forward(self,x):
		batch_size = x.size(0)
		x = F.relu(self.pooling(self.converlusion_1(x)))
		x = F.relu(self.pooling(self.converlusion_2(x)))
		x = x.view(batch_size,-1)
		x = self.fullconnect(x)
		return x

cnn_model = CNurelNet()

#3 construct loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn_model.parameters(),lr=0.1,momentum=0.5)

#4 training cycle(forward->(zero_grad)backward->step(update))
def train(epoch):
	running_loss = 0.0
	for batch_index,(inputs,target) in enumerate(train_loader,0):
		# 每次for循环都初始化一下梯度grad
		optimizer.zero_grad()

		outputs = cnn_model(inputs)
		# 得到预测值之后计算损失
		loss_val = criterion(outputs,target)

		# 反向传播自动计算tensor里面各个对象的导数
		loss_val.backward()
		# 用计算好的梯度更新权重参数W
		optimizer.step()

		running_loss += loss_val.item()

		if batch_index %300==299:
			print('[epoch:%d,batch_index:%5d] loss:%.3f' % (epoch+1,batch_index+1,running_loss/300))
			running_loss=0.0
	
def test():
	correct=0
	total = 0
	with torch.no_grad():
		for data in test_loader:
			(inputs,labels) = data
			outputs = cnn_model(inputs)  #outputs是每一个样本的所有类别预测的一个float值--每个float与一个label的one-hot对应(float越大越接近1的那个类是最后预测)
			max_index,pred = torch.max(outputs.data,dim=1)

			#计算acc(批次里面所有预测正确的总和除以批次的样本类别总和(累加的))
			correct += (pred==labels).sum().item()
			total += labels.size(0)
	print('accuracy on test set: %d %% ' % (100*correct/total))

for epoch in range(6):
	train(epoch)
	test()