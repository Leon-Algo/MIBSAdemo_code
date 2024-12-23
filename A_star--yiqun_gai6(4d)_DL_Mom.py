from ACO_fix_multifactor_MBGD_Momentum import ACO    ################################注意看这里用的第几代
from lddya.Draw import ShanGeTu,IterationGraph    #导入栅格图、迭代图的绘制模块
from lddya.Map import Map   #导入地图文件读取模块
import numpy as np
import time   #时间计算模块

# 开始计时：
time_start = time.perf_counter()

m = Map()    
m.load_map_file('华农光纤网map1.txt')    # 读取地图文件


file = open('./route_list_4diriction.txt', 'r')
route_load = []
for line in file.readlines():
    line = line.strip('\n')
    route_load.append(eval(line))
file.close()
print('读取初始路径成功！')     # 读取初始路径


with open('./站点载荷矩阵.txt','r') as f:
    a1 = f.readlines()
for i in range(len(a1)):
    a1[i] = list(a1[i].strip('\n'))
data1 = np.array(a1).astype('int64')

print('读取节点载荷成功！','\n')     # 读取节点载荷


aco = ACO(map_data=m.data,start=[7,0],end=[20,16],route_load=route_load,node_load=data1)      ############就是按照x,y坐标，之前被误导了，坐标不用反过来表示





# aco = ACO(map_data=m.data,start=[0,7],end=[15,19])    #初始化ACO，不调整任何参数的情况下，仅提供地图数据即可，本行中数据由Map.data提供，start跟end都是[y,x]格式，默认[0,0],[19,19]。


aco.run()                     # 迭代运行
sfig = ShanGeTu(map_data = m.data,start_point=[7,0],end_point=[20,16])       # 初始化栅格图绘制模块
# sfig.draw_way(aco.way_data_best)   # 绘制路径信息，路径数据由ACO.way_data_best提供。
ways_data = aco.ways_data
print('===========================================================================',len(ways_data),)
print(f"way_zuhe_best: {aco.way_zuhe_best}")

# colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0]]
# line_types = ['-', '-', '-', '-']
# sfig.draw_ways(ways_data, new_pic=True, colors=colors[:len(ways_data)], line_types=line_types[:len(ways_data)])

colors=[[174,35,48], [230,152,54], [246,222,107], [124,185,232]]##暖色调：可以使用深红和金黄色的组合来代表路径
# colors=[[0,70,173], [9,150,217], [184,228,228], [165,218,235]]#冷色调：可以使用深蓝和浅蓝的组合来代表路径
# colors=[[68,1,84], [145,36,80], [247,180,103], [253,231,146]]#地中海风格：可以使用紫罗兰色和浅黄色的组合来代表路径
# colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0]]
# colors=[(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
line_types=['-', '--', '-.', ':']
sfig.draw_ways(ways_data, new_pic=True, colors=colors[:len(ways_data)], line_types=line_types[:len(ways_data)])



sfig.save('aco+pher_init+pw(4D)111+h_nodeload+loadsum+node_num+(1-rou)+自适应MBGD路径图+Mom_MIBSA_1.jpg')                   #保存栅格图数据为'123.jpg'
dfig = IterationGraph(data_list= [aco.generation_aver,aco.generation_best],   #绘制数据: 每代平均、每代最优路径信息
                    style_list=['--r','-.g'],    # 线型 (规则同plt.plot()中的线型规则)
                    legend_list=['每代平均','每代最优'],  # 图例 (可选参数，可以不写)
                    xlabel='迭代次数',           # x轴标签，默认“x”
                    ylabel= '路径长度'           # y轴标签，默认“y”
                    )                 # 初始化迭代图绘制模块        
dfig.save('aco+pher_init+pw(4D)111+h_nodeload+loadsum+node_num+(1-rou)+自适应MBGD迭代图+Mom_MIBSA_1.jpg')                     #迭代图保存为321.jpg

# 结束计时：
time_end = time.perf_counter()
print("运行时间："+str((time_end - time_start))+"秒")