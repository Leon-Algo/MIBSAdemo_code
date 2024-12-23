import torch as tc
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt




'''1 prepare training dataset
2 design model using Class inherit from nn.Module
3 construct loss function and optimizer
4 training cycle(forward->(zero_grad)backward->step(update))'''

#1 prepare training dataset
x_data = tc.Tensor([[1.0],[2.0],[3.0]])
y_data = tc.Tensor([[0],[0],[1]])

#2 design model using Class inherit from nn.Module
class LogisticRegressionModel(tc.nn.Module):
	def __init__(self) -> None:
		super(LogisticRegressionModel,self).__init__()
		self.linear = tc.nn.Linear(1,1)
	
	def forward(self,x):
		y_pre = F.sigmoid(self.linear(x))
		return y_pre
logistic_model = LogisticRegressionModel()

#3 construct loss function and optimizer
criterion = tc.nn.BCELoss(size_average=False)
optimizer = tc.optim.SGD(logistic_model.parameters(),lr=0.1)

#4 training cycle(forward->(zero_grad)backward->step(update))
for epoch in range(100):
	y_pre = logistic_model(x_data)
	loss_val = criterion(y_pre,y_data)
	print(epoch,loss_val.item())

	optimizer.zero_grad()
	loss_val.backward()
	
	optimizer.step()

# test model --pridict  x=4
x_test = tc.Tensor([[4.0]])
y_predict = logistic_model(x_test)
print('y_predict=',y_predict.data)

# visualize pridiction curve
x_zhou = np.linspace(0,10,200)
x_test = tc.Tensor(x_zhou).view((200,1))
y_pred = logistic_model(x_test)
y_zhou = y_pred.data.numpy() 

plt.plot(x_zhou,y_zhou)
plt.plot([0,10],[0.5,0.5],c='r') #set a dividing line
plt.grid()
plt.show()