# import numpy as np
# import matplotlib.pyplot as plt


# # 建立“蚂蚁”类
# class Ant(object):
#     def __init__(self, path):
#         self.path = path                       # 蚂蚁当前迭代整体路径
#         self.length = self.calc_length(path)   # 蚂蚁当前迭代整体路径长度

#     def calc_length(self, path_):              # path=[A, B, C, D, A]注意路径闭环
#         length_ = 0
#         for i in range(len(path_)-1):
#             delta = (path_[i].x - path_[i+1].x, path_[i].y - path_[i+1].y)
#             length_ += np.linalg.norm(delta)
#         return length_

#     @staticmethod
#     def calc_len(A, B):                        # 静态方法，计算城市A与城市B之间的距离
#         return np.linalg.norm((A.x - B.x, A.y - B.y))


# # 建立“城市”类
# class City(object):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y


# # 建立“路径”类
# class Path(object):
#     def __init__(self, A):                     # A为起始城市
#         self.path = [A, A]

#     def add_path(self, B):                     # 追加路径信息，方便计算整体路径长度
#         self.path.append(B)
#         self.path[-1], self.path[-2] = self.path[-2], self.path[-1]


# # 构建“蚁群算法”的主体
# class ACO(object):
#     def __init__(self, ant_num=50, maxIter=300, alpha=1, beta=5, rho=0.1, Q=1):
#         self.ants_num = ant_num   # 蚂蚁个数
#         self.maxIter = maxIter    # 蚁群最大迭代次数
#         self.alpha = alpha        # 信息启发式因子
#         self.beta = beta          # 期望启发式因子
#         self.rho = rho            # 信息素挥发速度
#         self.Q = Q                # 信息素强度
#         ###########################
#         self.deal_data('coordinates.dat')                         # 提取所有城市的坐标信息
#         ###########################
#         self.path_seed = np.zeros(self.ants_num).astype(int)      # 记录一次迭代过程中每个蚂蚁的初始城市下标
#         self.ants_info = np.zeros((self.maxIter, self.ants_num))  # 记录每次迭代后所有蚂蚁的路径长度信息
#         self.best_path = np.zeros(self.maxIter)                   # 记录每次迭代后整个蚁群的“历史”最短路径长度
#         ###########################
#         self.solve()              # 完成算法的迭代更新
#         self.display()            # 数据可视化展示

#     def deal_data(self, filename):
#         with open(filename, 'rt') as f:
#             temp_list = list(line.split() for line in f)                                   # 临时存储提取出来的坐标信息
#         self.cities_num = len(temp_list)                                                   # 1. 获取城市个数
#         self.cities = list(City(float(item[0]), float(item[1])) for item in temp_list)     # 2. 构建城市列表
#         self.city_dist_mat = np.zeros((self.cities_num, self.cities_num))                  # 3. 构建城市距离矩阵
#         for i in range(self.cities_num):
#             A = self.cities[i]
#             for j in range(i, self.cities_num):
#                 B = self.cities[j]
#                 self.city_dist_mat[i][j] = self.city_dist_mat[j][i] = Ant.calc_len(A, B)
#         self.phero_mat = np.ones((self.cities_num, self.cities_num))                       # 4. 初始化信息素矩阵
#         # self.phero_upper_bound = self.phero_mat.max() * 1.2                              ###信息素浓度上限
#         self.eta_mat = 1/(self.city_dist_mat + np.diag([np.inf]*self.cities_num))          # 5. 初始化启发函数矩阵

#     def solve(self):
#         iterNum = 0                                                            # 当前迭代次数
#         while iterNum < self.maxIter:
#             self.random_seed()                                                 # 使整个蚁群产生随机的起始点
#             delta_phero_mat = np.zeros((self.cities_num, self.cities_num))     # 初始化每次迭代后信息素矩阵的增量
#             ##########################################################################
#             for i in range(self.ants_num):
#                 city_index1 = self.path_seed[i]                                # 每只蚂蚁访问的第一个城市下标
#                 ant_path = Path(self.cities[city_index1])                      # 记录每只蚂蚁访问过的城市
#                 tabu = [city_index1]                                           # 记录每只蚂蚁访问过的城市下标，禁忌城市下标列表
#                 non_tabu = list(set(range(self.cities_num)) - set(tabu))
#                 for j in range(self.cities_num-1):                             # 对余下的城市进行访问
#                     up_proba = np.zeros(self.cities_num-len(tabu))             # 初始化状态迁移概率的分子
#                     for k in range(self.cities_num-len(tabu)):
#                         up_proba[k] = np.power(self.phero_mat[city_index1][non_tabu[k]], self.alpha) * \
#                         np.power(self.eta_mat[city_index1][non_tabu[k]], self.beta)
#                     proba = up_proba/sum(up_proba)                             # 每条可能子路径上的状态迁移概率
#                     while True:                                                # 提取出下一个城市的下标
#                         random_num = np.random.rand()
#                         index_need = np.where(proba > random_num)[0]
#                         if len(index_need) > 0:
#                             city_index2 = non_tabu[index_need[0]]
#                             break
#                     ant_path.add_path(self.cities[city_index2])
#                     tabu.append(city_index2)
#                     non_tabu = list(set(range(self.cities_num)) - set(tabu))
#                     city_index1 = city_index2
#                 self.ants_info[iterNum][i] = Ant(ant_path.path).length
#                 if iterNum == 0 and i == 0:                                    # 完成对最佳路径城市的记录
#                     self.best_cities = ant_path.path
#                 else:
#                     if self.ants_info[iterNum][i] < Ant(self.best_cities).length: self.best_cities = ant_path.path
#                 tabu.append(tabu[0])                                           # 每次迭代完成后，使禁忌城市下标列表形成完整闭环
#                 for l in range(self.cities_num):
#                     delta_phero_mat[tabu[l]][tabu[l+1]] += self.Q/self.ants_info[iterNum][i]

#             self.best_path[iterNum] = Ant(self.best_cities).length

#             self.update_phero_mat(delta_phero_mat)                             # 更新信息素矩阵
#             iterNum += 1

#     def update_phero_mat(self, delta):
#         self.phero_mat = (1 - self.rho) * self.phero_mat + delta
#         # self.phero_mat = np.where(self.phero_mat > self.phero_upper_bound, self.phero_upper_bound, self.phero_mat) # 判断是否超过浓度上限

#     def random_seed(self):                                                     # 产生随机的起始点下表，尽量保证所有蚂蚁的起始点不同
#         if self.ants_num <= self.cities_num:                                   # 蚂蚁数 <= 城市数
#             self.path_seed[:] = np.random.permutation(range(self.cities_num))[:self.ants_num]
#         else:                                                                  # 蚂蚁数 > 城市数
#             self.path_seed[:self.cities_num] = np.random.permutation(range(self.cities_num))
#             temp_index = self.cities_num
#             while temp_index + self.cities_num <= self.ants_num:
#                 self.path_seed[temp_index:temp_index + self.cities_num] = np.random.permutation(range(self.cities_num))
#                 temp_index += self.cities_num
#             temp_left = self.ants_num % self.cities_num
#             if temp_left != 0:
#                 self.path_seed[temp_index:] = np.random.permutation(range(self.cities_num))[:temp_left]

#     def display(self):                                                         # 数据可视化展示
#         plt.figure(figsize=(6, 10))
#         plt.subplot(211)
#         plt.plot(self.ants_info, 'g.')
#         plt.plot(self.best_path, 'r-', label='history_best')
#         plt.xlabel('Iteration')
#         plt.ylabel('length')
#         plt.legend()
#         plt.subplot(212)
#         plt.plot(list(city.x for city in self.best_cities), list(city.y for city in self.best_cities), 'g-')
#         plt.plot(list(city.x for city in self.best_cities), list(city.y for city in self.best_cities), 'r.')
#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.savefig('ACO.png', dpi=500)
#         plt.show()
#         plt.close()



# 565.0   575.0
# 25.0   185.0
# 345.0   750.0
# 945.0   685.0
# 845.0   655.0
# 880.0   660.0
# 25.0    230.0
# 525.0   1000.0
# 580.0   1175.0
# 650.0   1130.0
# 1605.0   620.0
# 1220.0   580.0
# 1465.0   200.0
# 1530.0   5.0
# 845.0   680.0
# 725.0   370.0
# 145.0   665.0
# 415.0   635.0
# 510.0   875.0
# 560.0   365.0
# 300.0   465.0
# 520.0   585.0
# 480.0   415.0
# 835.0   625.0
# 975.0   580.0
# 1215.0   245.0
# 1320.0   315.0
# 1250.0   400.0
# 660.0   180.0
# 410.0   250.0
# 420.0   555.0
# 575.0   665.0
# 1150.0   1160.0
# 700.0   580.0
# 685.0   595.0
# 685.0   610.0
# 770.0   610.0
# 795.0   645.0
# 720.0   635.0
# 760.0   650.0
# 475.0   960.0
# 95.0   260.0
# 875.0   920.0
# 700.0   500.0
# 555.0   815.0
# 830.0   485.0
# 1170.0   65.0
# 830.0   610.0
# 605.0   625.0
# 595.0   360.0
# 1340.0   725.0
# 1740.0   245.0
# View coordinates.dat

# ACO()





# import numpy as np
# import matplotlib.pyplot as plt
 
 
# class ACO:
#     def __init__(self, parameters):
#         """
#         Ant Colony Optimization
#         parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
#         """
#         # 初始化
#         self.NGEN = parameters[0]  # 迭代的代数
#         self.pop_size = parameters[1]  # 种群大小
#         self.var_num = len(parameters[2])  # 变量个数
#         self.bound = []  # 变量的约束范围
#         self.bound.append(parameters[2])
#         self.bound.append(parameters[3])
 
#         self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有蚂蚁的位置
#         self.g_best = np.zeros((1, self.var_num))  # 全局蚂蚁最优的位置
 
#         # 初始化第0代初始全局最优解
#         temp = -1
#         for i in range(self.pop_size):
#             for j in range(self.var_num):
#                 self.pop_x[i][j] = np.random.uniform(self.bound[0][j], self.bound[1][j])
#             fit = self.fitness(self.pop_x[i])
#             if fit > temp:
#                 self.g_best = self.pop_x[i]
#                 temp = fit
 
#     def fitness(self, ind_var):
#         """
#         个体适应值计算
#         """
#         x1 = ind_var[0]
#         x2 = ind_var[1]
#         x3 = ind_var[2]
#         x4 = ind_var[3]
#         y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
#         return y
 
#     def update_operator(self, gen, t, t_max):
#         """
#         更新算子：根据概率更新下一时刻的位置
#         """
#         rou = 0.8   # 信息素挥发系数
#         Q = 1       # 信息释放总量
#         lamda = 1 / gen
#         pi = np.zeros(self.pop_size)
#         for i in range(self.pop_size):
#             for j in range(self.var_num):
#                 pi[i] = (t_max - t[i]) / t_max
#                 # 更新位置
#                 if pi[i] < np.random.uniform(0, 1):
#                     self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * lamda
#                 else:
#                     self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * (
#                                 self.bound[1][j] - self.bound[0][j]) / 2
#                 # 越界保护
#                 if self.pop_x[i][j] < self.bound[0][j]:
#                     self.pop_x[i][j] = self.bound[0][j]
#                 if self.pop_x[i][j] > self.bound[1][j]:
#                     self.pop_x[i][j] = self.bound[1][j]
#             # 更新t值
#             t[i] = (1 - rou) * t[i] + Q * self.fitness(self.pop_x[i])
#             # 更新全局最优值
#             if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
#                 self.g_best = self.pop_x[i]
#         t_max = np.max(t)
#         return t_max, t
 
#     def main(self):
#         popobj = []
#         best = np.zeros((1, self.var_num))[0]
#         for gen in range(1, self.NGEN + 1):
#             if gen == 1:
#                 tmax, t = self.update_operator(gen, np.array(list(map(self.fitness, self.pop_x))),
#                                      np.max(np.array(list(map(self.fitness, self.pop_x)))))
#             else:
#                tmax, t = self.update_operator(gen, t, tmax)
#             popobj.append(self.fitness(self.g_best))
#             print('############ Generation {} ############'.format(str(gen)))
#             print(self.g_best)
#             print(self.fitness(self.g_best))
#             if self.fitness(self.g_best) > self.fitness(best):
#                 best = self.g_best.copy()
#             print('最好的位置：{}'.format(best))
#             print('最大的函数值：{}'.format(self.fitness(best)))
#         print("---- End of (successful) Searching ----")
 
#         plt.figure()
#         plt.title("Figure1")
#         plt.xlabel("iterators", size=14)
#         plt.ylabel("fitness", size=14)
#         t = [t for t in range(1, self.NGEN + 1)]
#         plt.plot(t, popobj, color='b', linewidth=2)
#         plt.show()
 
 
# if __name__ == '__main__':
#     NGEN = 100
#     popsize = 100
#     low = [1, 1, 1, 1]
#     up = [30, 30, 30, 30]
#     parameters = [NGEN, popsize, low, up]
#     aco = ACO(parameters)
#     aco.main()



from lddya.Algorithm import ACO    #导入ACO算法
from lddya.Draw import ShanGeTu,IterationGraph    #导入栅格图、迭代图的绘制模块
from lddya.Map import Map   #导入地图文件读取模块
m = Map()    
m.load_map_file('华农光纤网map.txt')    # 读取地图文件
aco = ACO(map_data=m.data,start=[7,0],end=[19,15])      ############就是按照x,y坐标，之前被误导了，坐标不用反过来表示
# aco = ACO(map_data=m.data,start=[0,7],end=[15,19])    #初始化ACO，不调整任何参数的情况下，仅提供地图数据即可，本行中数据由Map.data提供，start跟end都是[y,x]格式，默认[0,0],[19,19]。
aco.run()                     # 迭代运行
sfig = ShanGeTu(map_data = m.data)       # 初始化栅格图绘制模块
sfig.draw_way(aco.way_data_best)   # 绘制路径信息，路径数据由ACO.way_data_best提供。
sfig.save('原始aco路径图.jpg')                   #保存栅格图数据为'123.jpg'
dfig = IterationGraph(data_list= [aco.generation_aver,aco.generation_best],   #绘制数据: 每代平均、每代最优路径信息
                    style_list=['--r','-.g'],    # 线型 (规则同plt.plot()中的线型规则)
                    legend_list=['每代平均','每代最优'],  # 图例 (可选参数，可以不写)
                    xlabel='迭代次数',           # x轴标签，默认“x”
                    ylabel= '路径长度'           # y轴标签，默认“y”
                    )                 # 初始化迭代图绘制模块        
dfig.save('原始aco迭代图.jpg')                     #迭代图保存为321.jpg




# from lddya.Algorithm import ACO    #导入ACO算法
# from lddya.Draw import ShanGeTu,IterationGraph    #导入栅格图、迭代图的绘制模块
# from lddya.Map import Map   #导入地图文件读取模块
# m = Map()    
# m.load_map_file('map.txt')    # 读取地图文件
# aco = ACO(map_data=m.data,start=[0,0],end=[19,19])    #初始化ACO，不调整任何参数的情况下，仅提供地图数据即可，本行中数据由Map.data提供，start跟end都是[y,x]格式，默认[0,0],[19,19]。
# aco.run()                     # 迭代运行
# sfig = ShanGeTu(map_data = m.data)       # 初始化栅格图绘制模块
# sfig.draw_way(aco.way_data_best)   # 绘制路径信息，路径数据由ACO.way_data_best提供。
# sfig.save('123.jpg')                   #保存栅格图数据为'123.jpg'
# dfig = IterationGraph(data_list= [aco.generation_aver,aco.generation_best],   #绘制数据: 每代平均、每代最优路径信息
#                     style_list=['--r','-.g'],    # 线型 (规则同plt.plot()中的线型规则)
#                     legend_list=['每代平均','每代最优'],  # 图例 (可选参数，可以不写)
#                     xlabel='迭代次数',           # x轴标签，默认“x”
#                     ylabel= '路径长度'           # y轴标签，默认“y”
#                     )                 # 初始化迭代图绘制模块        
# dfig.save('321.jpg')                     #迭代图保存为321.jpg


# from lddya.Algorithm import ACO    #导入ACO算法
# from lddya.Draw import ShanGeTu,IterationGraph    #导入栅格图、迭代图的绘制模块
# from lddya.Map import Map   #导入地图文件读取模块
# m = Map()    
# m.load_map_file('简单map.txt')    # 读取地图文件
# aco = ACO(map_data=m.data,start=[0,0],end=[2,0],max_iter=1,ant_num=5,pher_init=4,evaporate=0.5)   
# # aco = ACO(map_data=m.data,start=[0,0],end=[2,0])      ############就是按照x,y坐标，之前被误导了，坐标不用反过来表示
# # aco = ACO(map_data=m.data,start=[0,7],end=[15,19])    #初始化ACO，不调整任何参数的情况下，仅提供地图数据即可，本行中数据由Map.data提供，start跟end都是[y,x]格式，默认[0,0],[19,19]。
# aco.run()                     # 迭代运行
# sfig = ShanGeTu(map_data = m.data)       # 初始化栅格图绘制模块
# sfig.draw_way(aco.way_data_best)   # 绘制路径信息，路径数据由ACO.way_data_best提供。
# sfig.save('3_1.jpg')                   #保存栅格图数据为'123.jpg'
# dfig = IterationGraph(data_list= [aco.generation_aver,aco.generation_best],   #绘制数据: 每代平均、每代最优路径信息
#                     style_list=['--r','-.g'],    # 线型 (规则同plt.plot()中的线型规则)
#                     legend_list=['每代平均','每代最优'],  # 图例 (可选参数，可以不写)
#                     xlabel='迭代次数',           # x轴标签，默认“x”
#                     ylabel= '路径长度'           # y轴标签，默认“y”
#                     )                 # 初始化迭代图绘制模块        
# dfig.save('3_2.jpg')                     #迭代图保存为321.jpg