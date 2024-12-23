
import numpy as np
import torch
import matplotlib.pyplot as plt

'''1 prepare training dataset
2 design model using Class inherit from nn.Module
3 construct loss function and optimizer
4 training cycle(forward->(zero_grad)backward->step(update))'''

#1 prepare training dataset
x_y =  np.loadtxt('D:\\jupyter_code\\刘二PyTorch深度学习实践\\diabetes.csv.gz',delimiter = ',',dtype = np.float32)
x_data = torch.from_numpy(x_y[:,:-1])
y_data = torch.from_numpy(x_y[:,[-1]])  #[0]是为了防止这单独一列变成向量，加[]后则以矩阵存在

#2 design model using Class inherit from nn.Module
class artificial_NN(torch.nn.Module):
	def __init__(self) -> None:
		super(artificial_NN,self).__init__()
		self.linear1 = torch.nn.Linear(8,6)
		self.linear2 = torch.nn.Linear(6,4)
		self.linear3 = torch.nn.Linear(4,1)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self,x):
		x = self.sigmoid(self.linear1(x))
		x = self.sigmoid(self.linear2(x))
		x = self.sigmoid(self.linear3(x))
		return x
artificial_nn = artificial_NN()

#3 construct loss function and optimizer
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(artificial_nn.parameters(),lr=0.1)

#4 training cycle(forward->(zero_grad)backward->step(update))
epoch_list = []
loss_list = []
for epoch in range(100):
	#forward算loss
	y_pred = artificial_nn(x_data)
	loss_val = criterion(y_pred,y_data)
	epoch_list.append(epoch)
	loss_list.append(loss_val.item())

	#注意先grad清零，再backward计算grad（存档）
	optimizer.zero_grad()
	loss_val.backward()

	#用算好的grad来update每一个weight（或者之间x)
	optimizer.step()

#loss变化的可视化
plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#看看每一layer的参数
layer1_weight = artificial_nn.linear1.weight.data
layer1_bais = artificial_nn.linear1.bias.data
print('layer1_weight:',layer1_weight,'shape:',layer1_weight.shape)
print('layer1_bais:',layer1_bais,'shape:',layer1_bais.shape)

