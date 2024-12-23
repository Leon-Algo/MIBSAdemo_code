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
transform_operation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])  #系列转化操作--转tensor，标准化

train_data = datasets.MNIST(root='./dataset/mnist/',train=True,download=False,transform=transform_operation)
train_loader = DataLoader(train_data,shuffle=True,batch_size=batch_size)

test_data = datasets.MNIST(root='./dataset/mnist/',train=True,download=False,transform=transform_operation)
test_loader = DataLoader(test_data,shuffle=True,batch_size=batch_size)


#2 design model using Class inherit from nn.Module
#Inception块--同时并行训练1x1，3x3，5x5，avgpool_1x1层，特征提取能力好的卷积层自然其权重就会更大
class Inception1(torch.nn.Module):
	def __init__(self,in_channels) -> None:
		super(Inception1,self).__init__()

		self.branch1x1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)

		self.branch3x3_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
		self.branch3x3_2 = torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
		self.branch3x3_3 = torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

		self.branch5x5_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
		self.branch5x5_2 = torch.nn.Conv2d(16,24,kernel_size=5,padding=2)

		self.branch_pool1x1 = torch.nn.Conv2d(in_channels,24,kernel_size=1)

	def forward(self,x):
		branch1x1 = self.branch1x1(x)

		branch5x5 = self.branch5x5_1(x)
		branch5x5 = self.branch5x5_2(branch5x5)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)
		branch3x3 = self.branch3x3_3(branch3x3)

		branch_pool1x1 = F.avg_pool2d(x,kernel_size=3,padding=1,stride=1)
		branch_pool1x1 = self.branch_pool1x1(branch_pool1x1)

		output_inception1 = [branch_pool1x1,branch1x1,branch3x3,branch5x5]
		return torch.cat(output_inception1,dim=1)  # 原始tensor维度顺序为b,c,w,h。dim=1表示按c维度(channel)拼接

#加个残差层(跳连接层)可以有效减小梯度消失问题(这里采用最原始的残差层模型--还有很多的变体)
# class Residuel_net1(torch.nn.Module):
# 	def __init__(self,in_channels) -> None:
# 		super(Residuel_net1,self).__init__()
# 		self.in_channels = in_channels
# 		self.conv1 = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
# 		self.conv2 = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)

# 	def forward(self,x):
# 		y = F.relu(self.conv1(x))
# 		y = self.conv2(y)
# 		return F.relu(x+y)

#将inception1模块和residualNet模块堆砌起来，组成新的CNN网络
class Cnn_2(torch.nn.Module):
	def __init__(self) -> None:
		super(Cnn_2,self).__init__()
		self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
		
		self.incep1 = Inception1(in_channels=10)  #对应于conv1中的输出通道为10，作为输入通道数

		self.mp = torch.nn.MaxPool2d(2)

		self.conv2 = torch.nn.Conv2d(88,20,kernel_size=5)	

		# self.rblock = Residuel_net1(16)

		self.incep2 = Inception1(in_channels=20)

		# self.conv3 = torch.nn.Conv2d(88,32,kernel_size=1)

		self.fullconnect = torch.nn.Linear(1408,10)

	def forward(self,x):
		in_size = x.size(0)

		x = self.mp(F.relu(self.conv1(x)))
		x = self.incep1(x)
		x = self.mp(F.relu(self.conv2(x)))
		# x = self.rblock(x)
		x = self.incep2(x)
		# x = self.mp(F.relu(self.conv3(x)))
		x = x.view(in_size,-1)
		x = self.fullconnect(x)

		return x 

dajimo_nn = Cnn_2()

#3 construct loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(dajimo_nn.parameters(),lr=0.01,momentum=0.5)

#4 training cycle(forward->(zero_grad)backward->step(update))
def train(epoch):
	running_loss =0.0
	for batch_index,data in enumerate(train_loader,0):
		inputs,target = data
		#每次for循环(batch)都要把gard初始化先
		optimizer.zero_grad()

		outputs = dajimo_nn(inputs)
		loss = criterion(outputs,target)
		loss.backward()
		optimizer.step()

		running_loss = loss.item()

		if batch_index %300 ==299:
			print('epoch:%d,batch_index:%5d,loss:%.3f' % (epoch+1,batch_index+1,running_loss/300))
			running_loss=0.0
			

def test():
	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_loader:
			images,labels = data
			outputs = dajimo_nn(images)
			max_index,predicted = torch.max(outputs.data,dim=1)

			correct += (predicted == labels).sum().item()
			total += labels.size(0)
	print("accuracy on test set:%d,"%(100*correct/total))

for epoch in range(3):
	train(epoch)
	test()	


