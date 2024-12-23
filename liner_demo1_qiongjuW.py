import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl 



'''用穷举法来更新W——线性回归'''

# 给一组数据
x_data = [1,2,3,4,5]
y_data = [2,4,6,8,10]

def forward_preprogettion(x,w,b):
	y_pre0 = x*w+b
	return y_pre0

def loss(y,y_pre):
	return (y_pre-y)**2

def qiongju_w():
	w_list=[]
	b_list=[]
	mse_list=[]
	for w in np.arange(0.0,4.0,0.1):
		for b in np.arange(-2.0,3.0,0.1):
			# print(f'穷举权重w为{w}的情况')
			l_sum = 0
			for x,y in zip(x_data,y_data):
				y_pre = forward_preprogettion(x=x,w=w,b=b)
				l_sum+= loss(y=y,y_pre=y_pre)
			mse = l_sum/len(x_data)
			# print('均方损失为MSE\n',mse)
			b_list.append(b)
			w_list.append(w)
			mse_list.append(mse)
	# plt.figure(figsize=(20,8),dpi=100)
	# plt.plot(w_list,mse_list)
	# plt.show()

	# fig = plt.figure()  #定义新的三维坐标轴
	# ax3 = plt.axes(projection='3d')
	# ax3.plot_surface(w_list,b_list,mse_list,cmap='rainbow')
	# plt.show()
	w_nd = np.array(w_list)
	b_nd = np.array(b_list)
	mse_nd = np.array(mse_list)
	# fig = plt.figure() 
	# ax = fig.add_subplot(111, projection='3d') 
	# ax.plot_surface(w_nd,b_nd,mse_nd, color='b') 
	# plt.show()

	mpl.rcParams['legend.fontsize'] = 10
	fig = plt.figure() 
	ax = fig.gca(projection='3d') 
	ax.plot(w_nd,b_nd,mse_nd,label='parametric curve') 
	ax.legend() 
	plt.show()


if __name__=='__main__':	

	liner_demo=qiongju_w()
	# print('\n',liner_demo)
