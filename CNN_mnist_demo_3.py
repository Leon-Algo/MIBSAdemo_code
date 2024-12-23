import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 需要F中的激活函数RULE，均值池化层avg_pool2d，

'''
1 prepare training dataset
2 design model using Class inherit from nn.Module
3 construct loss function and optimizer
4 training cycle(forward->(zero_grad)backward->step(update))
5 visualize
'''

#1 prepare training dataset
batch_size = 64
transform_operation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])

train_data = datasets.MNIST(root='./dataset/mnist/',train=True,download=False,transform=transform_operation)
train_loader = DataLoader(train_data,shuffle=True,batch_size=batch_size)

test_data = datasets.MNIST(root='./dataset/mnist/',train=True,download=False,transform=transform_operation)
test_loader = DataLoader(test_data,shuffle=True,batch_size=batch_size)

#2 加个残差层(跳连接层)可以有效减小梯度消失问题(这里采用最原始的残差层模型--还有很多的变体)
class Residuel_net1(torch.nn.Module):
	def __init__(self,in_channels) -> None:
		super(Residuel_net1,self).__init__()
		self.in_channels = in_channels
		self.conv1 = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
		self.conv2 = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)

	def forward(self,x):
		y = F.relu(self.conv1(x))
		y = self.conv2(y)
		return F.relu(x+y)

class Cnn_3(torch.nn.Module):
	def __init__(self) -> None:
		super(Cnn_3,self).__init__()
		self.conv1 = torch.nn.Conv2d(1,16,kernel_size=5)
		self.conv2 = torch.nn.Conv2d(16,32,kernel_size=5)

		self.rblock1 = Residuel_net1(16)
		self.rblock2 = Residuel_net1(32)

		self.mp = torch.nn.MaxPool2d(2)
		self.fc = torch.nn.Linear(512,10)

	def forward(self,x):
		in_size = x.size(0)

		x = self.mp(F.relu(self.conv1(x)))
		x = self.rblock1(x)
		x = self.mp(F.relu(self.conv2(x)))
		x = self.rblock2(x)

		x = x.view(in_size,-1)
		x = self.fc(x)

		return x 
	
dajimo_resnn = Cnn_3()

#3 construct loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(dajimo_resnn.parameters(),lr=0.01,momentum=0.5)

#4 training cycle(forward->(zero_grad)backward->step(update))
def train(epoch):
	running_loss = 0.0
	for batch_index,(inputs,target) in enumerate(train_loader,0):
		optimizer.zero_grad()

		outputs = dajimo_resnn(inputs)
		loss = criterion(outputs,target)
		loss.backward()  #反馈传播计算梯度
		optimizer.step() #利用计算好的梯度更新权重参数

		running_loss += loss.item()
		if batch_index % 300 ==299:
			print('epoch:%d,batch_index:%5d,loss:%.3f' % (epoch+1,batch_index+1,running_loss/300))
			running_loss=0.0

def test():
	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_loader:
			images,labels = data
			outputs = dajimo_resnn(images)
			_,predicted = torch.max(outputs.data,dim=1)
			total += labels.size(0)
			correct += (predicted==labels).sum().item()
	print("accuracy on test set:%d,"%(100*correct/total))

for epoch in range(3):
	train(epoch)
	test()	