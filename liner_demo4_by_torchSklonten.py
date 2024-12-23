
import torch as tc
import numpy as np

from liner_demo3_torchSGD import loss


'''基本套路就是
1、将数据装入tensor(都是矩阵形式存放)
2、设计模型,这里面init和forward是必须重写的,其他的backward,step等等都是可以直接调用api的
3、构建损失函数和优化器(可以直接调用torch各种api,或者优化器直接用Adam)
4、可以循环训练了(forward,backward,step)'''

# prepare dataset
x_data = tc.Tensor([[1.0],[2.0],[3.0]])
y_data = tc.Tensor([[2.0],[4.0],[6.0]])

#design model by using torch inherit from nn.model
class linearmodel(tc.nn.Module):
	def __init__(self) -> None:
		super(linearmodel,self).__init__()
		self.linear = tc.nn.Linear(1,1)

	def forward(self,x):
		y_pre = self.linear(x)
		return y_pre
#实例化刚刚设计的方法
model = linearmodel()

#create our loss function and optimizer (or we can use api of torch derictly)
criterion = tc.nn.MSELoss(size_average=False)
optimizer = tc.optim.SGD(model.parameters(),lr=0.05)  #这里的tc.optim.SGD也是一个类，这里继承其父类model的parameters方法(可自动的选出设置有权重的tensor)

#training cycle
for epoch in range(100):
	y_pre = model(x_data)
	loss_val = criterion(y_pre,y_data)
	print(epoch,loss_val.item())

	optimizer.zero_grad()
	loss_val.backward()
	optimizer.step()

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

# test model --pridict  x=4
x_test = tc.Tensor([[4.0]])
y_predict = model(x_test)
print('y_predict=',y_predict.data)