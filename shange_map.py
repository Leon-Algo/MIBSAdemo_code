# from lddya.Algorithm import ACO    #导入ACO算法

# from ACO_fix_multifactor_MBGD import ACO
from lddya.Draw import ShanGeTu,IterationGraph    #导入栅格图、迭代图的绘制模块
from lddya.Map import Map   #导入地图文件读取模块
import numpy as np

m = Map()    
m.load_map_file('华农光纤网map1.txt')    # 读取地图文件

sfig = ShanGeTu(map_data = m.data)       # 初始化栅格图绘制模块
sfig.save('栅格地图建模.jpg')

# file = open('./route_list_4diriction.txt', 'r')
# route_load = []
# for line in file.readlines():
#     line = line.strip('\n')
#     route_load.append(eval(line))
# file.close()
# print('读取初始路径成功！')     # 读取初始路径


# with open('./站点载荷矩阵.txt','r') as f:
#     a1 = f.readlines()
# for i in range(len(a1)):
#     a1[i] = list(a1[i].strip('\n'))
# data1 = np.array(a1).astype('int64')

# print('读取节点载荷成功！','\n')     # 读取节点载荷


# aco = ACO(map_data=m.data,start=[7,0],end=[20,16],route_load=route_load,node_load=data1)      ############就是按照x,y坐标，之前被误导了，坐标不用反过来表示





# aco = ACO(map_data=m.data,start=[0,7],end=[15,19])    #初始化ACO，不调整任何参数的情况下，仅提供地图数据即可，本行中数据由Map.data提供，start跟end都是[y,x]格式，默认[0,0],[19,19]。
# aco.run()                     # 迭代运行
# sfig = ShanGeTu(map_data = m.data)       # 初始化栅格图绘制模块
# sfig.save('栅格地图建模.jpg')