import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split

'''
1 prepare training dataset
2 design model using Class inherit from nn.Module
3 construct loss function and optimizer
4 training cycle(forward->(zero_grad)backward->step(update))
'''

#1 prepare training dataset
class DiabetesDataset(Dataset):
	def __init__(self,filepath) -> None:
		self.row_date = np.loadtxt(filepath,delimiter = ',',dtype = np.float32)
		self.len = self.row_date.shape[0]  #返回样本的个数
		self.x_data = torch.from_numpy(self.row_date[:,:-1])
		self.y_data = torch.from_numpy(self.row_date[:,[-1]])
	
	def __getitem__(self, index):
		return self.x_data[index],self.y_data[index]

	def __len__(self):
		return self.len

	def split_data(self):
		x_train,x_test,y_train,y_test = train_test_split(self.row_date[:,:-1],self.row_date[:,[-1]],train_size=0.8)
		xtest = torch.from_numpy(x_test)
		ytest = torch.from_numpy(y_test)
		return xtest,ytest

dataest = DiabetesDataset('D:\\jupyter_code\\刘二PyTorch深度学习实践\\diabetes.csv.gz')
train_loader = DataLoader(dataset=dataest,batch_size=32,shuffle=True)

#2 design model using Class inherit from nn.Module
class Artificial_NN(torch.nn.Module):
	def __init__(self) -> None:
		super(Artificial_NN,self).__init__()
		self.linear1 = torch.nn.Linear(8,6)
		self.linear2 = torch.nn.Linear(6,4)
		self.linear3 = torch.nn.Linear(4,1)	
		self.activator = torch.nn.Sigmoid()

	def forward(self,x):
		x = self.activator(self.linear1(x))
		x = self.activator(self.linear2(x))
		x = self.activator(self.linear3(x))
		return x
artificial_nn = Artificial_NN()

#3 construct loss function and optimizer
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(artificial_nn.parameters(),lr=0.1)

#4 training cycle(forward->(zero_grad)backward->step(update))
def train_Cycle(epoch):
	train_loss = 0.0
	count = 0
	for i,(inputs,labels) in enumerate(train_loader,0):
		y_pred = artificial_nn(inputs)
		loss_val = criterion(y_pred,labels)
		train_loss += loss_val.item()
		count = i

		optimizer.zero_grad()
		loss_val.backward()

		optimizer.step()

	train_loss_val = train_loss/count
	return train_loss_val

def loss_decent_visualize(epoch_list,loss_list):
	plt.plot(epoch_list,loss_list)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()

def test():
	with torch.no_grad():
		xtest,ytest = dataest.split_data()
		y_pred = artificial_nn(xtest)
		y_pred_labels = torch.where(y_pred>=0.5,torch.Tensor([1.0]),torch.Tensor([0.0]))
		acc = torch.eq(y_pred_labels,ytest).sum().item()/ytest.size(0)
		print('test acc:',acc)

if __name__ == '__main__':
	epoch_list = []
	loss_list = []
	for epoch in range(1000):
		train_loss_val = train_Cycle(epoch)

		loss_list.append(train_loss_val)
		epoch_list.append(epoch)


		if epoch%100==99:
			test()

	loss_decent_visualize(epoch_list,loss_list)
