# from lddya.Algorithm import ACO    #导入ACO算法

from ACO_fix import ACO
from lddya.Draw import ShanGeTu,IterationGraph    #导入栅格图、迭代图的绘制模块
from lddya.Map import Map   #导入地图文件读取模块
m = Map()    
m.load_map_file('华农光纤网map.txt')    # 读取地图文件

file = open('./route_list.txt', 'r')
route_load = []
for line in file.readlines():
    line = line.strip('\n')
    route_load.append(eval(line))
file.close()
print('读取初始路径成功！')     # 读取初始路径

aco = ACO(map_data=m.data,start=[7,0],end=[19,15],route_load=route_load)      ############就是按照x,y坐标，之前被误导了，坐标不用反过来表示





# aco = ACO(map_data=m.data,start=[0,7],end=[15,19])    #初始化ACO，不调整任何参数的情况下，仅提供地图数据即可，本行中数据由Map.data提供，start跟end都是[y,x]格式，默认[0,0],[19,19]。
aco.run()                     # 迭代运行
sfig = ShanGeTu(map_data = m.data)       # 初始化栅格图绘制模块
sfig.draw_way(aco.way_data_best)   # 绘制路径信息，路径数据由ACO.way_data_best提供。
sfig.save('aco+pher_init路径图.jpg')                   #保存栅格图数据为'123.jpg'
dfig = IterationGraph(data_list= [aco.generation_aver,aco.generation_best],   #绘制数据: 每代平均、每代最优路径信息
                    style_list=['--r','-.g'],    # 线型 (规则同plt.plot()中的线型规则)
                    legend_list=['每代平均','每代最优'],  # 图例 (可选参数，可以不写)
                    xlabel='迭代次数',           # x轴标签，默认“x”
                    ylabel= '路径长度'           # y轴标签，默认“y”
                    )                 # 初始化迭代图绘制模块        
dfig.save('aco+pher_init迭代图.jpg')                     #迭代图保存为321.jpg