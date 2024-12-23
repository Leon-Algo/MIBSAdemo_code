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
batch_size = 64 #全局变量
transform_operation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]) 
			#对图像像素转化为张量，顺便标准化，把像素从0~255---神经网络更喜欢-1~1内正态分布的数据，类似范围的一个数据分布也是最有帮助的
train_dataset = datasets.MNIST(root='./dataset/mnist/',train=True,download=False,transform=transform_operation)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

test_dataset = datasets.MNIST(root='./dataset/mnist/',train=False,download=False,transform=transform_operation)
test_loader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size)



#2 design model using Class inherit from nn.Module
class ANN_multiclassfication(torch.nn.Module):
	def __init__(self) -> None:
		super(ANN_multiclassfication,self).__init__()
		self.linear1 = torch.nn.Linear(784,512)
		self.linear2 = torch.nn.Linear(512,256)
		self.linear3 = torch.nn.Linear(256,128)
		self.linear4 = torch.nn.Linear(128,64)
		self.linear5 = torch.nn.Linear(64,10)

	def forward(self,x):
		x = x.view(-1,784)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = F.relu(self.linear3(x))
		x = F.relu(self.linear4(x))
		x = self.linear5(x)  #如果调用nn.CrossEntropyLoss()就不需要对最后一个隐藏层做激活，包里面会自动的调用softmax做激活
		return x
ann = ANN_multiclassfication()

#3 construct loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ann.parameters(),lr=0.1,momentum=0.5)

#4 training cycle(forward->(zero_grad)backward->step(update))
def train_cycle(epoch):
	batch_index_list = []
	loss_list = []
	running_train_loss = 0.0
	for batch_index,(inputs,target) in enumerate(train_loader,0):
		optimizer.zero_grad()

		outputs = ann(inputs)   #模型预测的输出大小为(64 样本,10 分类)

		loss_val = criterion(outputs,target)  #target大小为(10) 标签y的类型是LongTensor

		loss_val.backward()
		optimizer.step()

		running_train_loss += loss_val.item()

		#每训练300个样本(batch_index=300)时显示一下loss_val
		if batch_index%100==99:
			ave_loss = running_train_loss/100
			print('%d,%5d,loss:%.3f' % (epoch+1,batch_index+1,ave_loss))
			loss_list.append(ave_loss)   ###这里有问题？？？为什么不是按输出的（27个保存），却只保存了9个loss
			running_train_loss = 0.0
	return loss_list

def test():
	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_loader:
			(inputs,labels) = data
			outputs = ann(inputs)  #outputs是每一个样本的所有类别预测的一个float值--每个float与一个label的one-hot对应(float越大越接近1的那个类是最后预测)
			max_index,pred = torch.max(outputs.data,dim=1)

			#计算acc(批次里面所有预测正确的总和除以批次的样本类别总和(累加的))
			correct += (pred==labels).sum().item()
			total += labels.size(0)
	print('accuracy on test set: %d %% ' % (100*correct/total))

def loss_decent_visualize(loss_list):
	plt.plot(list(np.arange(0,len(loss_list))),loss_list)
	plt.xlabel('num')
	plt.ylabel('loss')
	plt.show()

# if __name__ == '__mian__':

for epoch in range(3):
	loss_list = train_cycle(epoch)

	loss_list += loss_list
	test()

# loss_decent_visualize(loss_list)

# train_cycle(1)	