import numpy as np
import pandas as pd
import copy
from collections import Counter
################################################## 1 蚁群算法路径规划 ###########################################


# Ant只管通过地图数据以及信息素数据，输出一条路径。其他的你不用管。
class Ant():
    def __init__(self,node_load,start,end,max_step,pher_imp,dis_imp) -> None:
        self.max_step = max_step    # 蚂蚁最大行动力
        self.pher_imp = pher_imp    # 信息素重要性系数
        self.dis_imp = dis_imp      # 距离重要性系数
        self.start = start          # 蚂蚁初始位置[y,x] = [0,0],考虑到列表索引的特殊性，先定y，后定x
        self.destination = end  # 默认的终点节点(在run方法中会重新定义该值)
        self.successful = True      #标志蚂蚁是否成功抵达终点
        self.record_way = [start]   #路径节点信息记录
        self.node_load = node_load  #光标节点载荷矩阵

    def run(self,map_data,pher_data):         ############这就是一个走N步的函数(非核心)
        self.position = copy.deepcopy(self.start)
        #Step 1:不断找下一节点，直到走到终点或者力竭 
        for i in range(self.max_step):
            r = self.select_next_node(map_data,pher_data)  #不停的找下一个节点（直到精疲力竭）
            if r == False:        #判断寻找是否出现错误
                self.successful = False
                break
            else:
                if self.position == self.destination:      ##########################找到终点（successful默认就是True，不需再改了）
                    break
        else:                  #########else属于循环体语句，执行break就不执行这个else，否则可以无视else作用
            self.successful = False
    
    def select_next_node(self,map_data,pher_data):
        '''
        Function:
        ---------
        选择下一节点，结果直接存入self.postion，仅返回一个状态码True/False标志选择的成功与否。
        '''
        y_1 = self.position[0]
        x_1 = self.position[1]
        #Step 1:计算理论上的周围节点                                ###栅格图结构，左上为起点
        # node_be_selected = [[y_1-1,x_1-1],[y_1-1,x_1],[y_1-1,x_1+1],     #上一层
        #                     [y_1,x_1-1],              [y_1,x_1+1],       #同层
        #                     [y_1+1,x_1-1],[y_1+1,x_1],[y_1+1,x_1+1],     #下一层
        #                 ]             ##############注意node_be_selected是包含了可行点和非法点在内的全部周围节点

        #改为四方向搜索
        node_be_selected = [            [y_1-1,x_1],     #上一层
                            [y_1,x_1-1],              [y_1,x_1+1],       #同层
                                        [y_1+1,x_1],   #下一层
                        ]             ##############注意node_be_selected是包含了可行点和非法点在内的全部周围节点

        #Step 2:排除非法以及障碍物节点    
        node_be_selected_1 = []              ##################node_be_selected_1是在前者基础上排除非法点
        for i in node_be_selected:
            if i[0]<0 or i[1]<0:       #######非法负索引点排除
                continue
            if i[0]>=len(map_data) or i[1]>=len(map_data):   ########非法外边界点排除
                continue
            if map_data[i[0]][i[1]] == 0:           ################添加可行点
                node_be_selected_1.append(i)
        if len(node_be_selected_1) == 0:    # 如果无合法节点，则直接终止节点的选择
            return False
        if self.destination in node_be_selected_1:   # 如果到达终点旁，则直接选中终点
            self.position = self.destination
            self.record_way.append(copy.deepcopy(self.position))
            map_data[self.position[0]][self.position[1]] = 1
            return True
        #Step 3:计算节点与终点之间的距离，构建距离启发因子
        dis_1 = []    # 距离启发因子#######################################################################################
        for i in node_be_selected_1:                                                       #######1/(d+z)加入距离和载荷作为可见度的代价#################################
            dis_1.append(((self.destination[0]-i[0])**2+(self.destination[1]-i[1])**2)**0.5 + self.node_load[i[0]][i[1]])   ###############
        #Step 3.1:倒数反转
        for j in range(len(dis_1)):
            dis_1[j] = 1/dis_1[j]

        #Step 4:计算节点被选中的概率
        prob = []
        for i in range(len(node_be_selected_1)):           #############len(map_data)为栅格图的规模数---方阵行数
            p = (dis_1[i]**self.dis_imp) * (pher_data[y_1*len(map_data)+x_1][node_be_selected_1[i][0]*len(map_data)+node_be_selected_1[i][1]]**self.pher_imp)
            prob.append(p)                                ############就把这里的p看作是概率的分子部分得了
        #Step 5:轮盘赌选择某节点                           ####################这里由于下面的信息素矩阵维度为平方，所以这里也倍乘了一个map_lenght
        prob_sum = sum(prob)
        for i in range(len(prob)):
            prob[i] = prob[i]/prob_sum
        rand_key = np.random.rand()
        for k,i in enumerate(prob):
            if rand_key<=i:    ###########因为rand_key是一个均匀分布随机值，所以当哪个节点转移概率越大就越容易在那个节点终止循环，即索引k选择那个节点
                break
            else:
                rand_key -= i          ##########这里这个值无意义
        #Step 6:更新当前位置，并记录新的位置，将之前的位置标记为不可通过
        self.position = copy.deepcopy(node_be_selected_1[k])
        self.record_way.append(copy.deepcopy(self.position))           #######record_way是一个记录每只蚂蚁成功路径的列表---每次append一个可行点
        map_data[self.position[0]][self.position[1]] = 1             ######选择完了的节点标为不可通过----更新了禁忌表
        return True

class ACO():
    def __init__(self,map_data,route_load,node_load,start = [0,0],end = [19,19],tao_max = 10,tao_min=0.0001,init_pher_pow=6,max_iter = 100,ant_num = 50,pher_imp = 1,dis_imp = 10,evaporate_init = 0.9,) -> None:
        '''
            Params:
            --------
                pher_imp : 信息素重要性系数
                dis_imp  : 距离重要性系数
                evaporate: 信息素挥发系数(指保留的部分)
                pher_init: 初始信息素浓度
        '''
        #Step 0: 参数定义及赋值
        self.max_iter = max_iter       #最大迭代次数
        self.ant_num  = ant_num        #蚂蚁数量
        self.init_pher_pow = init_pher_pow ####初试路径信息素倍数
        self.ant_gener_pher = self.init_pher_pow    #每只蚂蚁携带的最大信息素总量

        self.tao_max = init_pher_pow*1.5   #信息素最大上限值
        self.tao_min = 0.1/self.max_iter    #信息素最小下限值

        # self.pher_init = pher_init #初始信息素浓度

        self.ant_params = {        #生成蚂蚁时所需的参数
            'dis_imp':dis_imp,
            'pher_imp': pher_imp,
            'start'   : start,
            'end'     : end
        }
        self.map_data = map_data.copy()        #地图数据
        self.map_lenght = self.map_data.shape[0]  #地图尺寸,用来标定蚂蚁的最大体力           ########地图尺寸的平方--因为路径上的信息素具有方向性
        # self.pher_data = pher_init*np.ones(shape=[self.map_lenght*self.map_lenght, #################################################
        #                                     self.map_lenght*self.map_lenght])    ##################信息素矩阵################（方阵）！！

        self.node_load = node_load



##############################################333
        # self.route_load = route_load.copy()

        # self.pher_data = self.init_pher_pow*np.ones(shape=[self.map_lenght*self.map_lenght, #################################################
        # self.map_lenght*self.map_lenght])    ##################信息素矩阵################（方阵）！！




#  初始信息素指数扩散衰减的计算方式
#         pher_data_init = np.ones(shape=[self.map_lenght**2,self.map_lenght**2])
#         j_3 = np.array(self.route_load)

#         # 计算每个点到路径上所有点的曼哈顿距离
#         dist_to_route = np.zeros((self.map_lenght**2, len(j_3)))
#         for i in range(self.map_lenght**2):
#             for j in range(len(j_3)):
#                 dist_to_route[i,j] = abs(i//self.map_lenght-j_3[j][0]) + abs(i%self.map_lenght-j_3[j][1])

#         # 对信息素矩阵进行初始化
#         for i in range(self.map_lenght**2):
#             for j in range(self.map_lenght**2):
#                 dist = min(dist_to_route[i])+min(dist_to_route[j])  # 点 i 和点 j 距离路径上的最小曼哈顿距离和
#                 # if j_3[-1][0]*self.map_lenght+j_3[-1][1]==j:  # 如果点 j 是路径的终点，则其信息素浓度设为 0
#                 #     pher_data_init[i][j] = 0
#                 # else:  # 否则，依次递减
#                 #     pher_data_init[i][j] = (1/self.init_pher_pow)**(dist/10)  # dist/10 是根据经验调整的因子，可以根据实际情况调整
#                 pher_data_init[i][j] = (1/self.init_pher_pow)**(dist/10)
#         self.pher_data = pher_data_init






        self.pher_decay_rate=0.05
        self.route_load = route_load.copy()  
        pher_data_init = np.zeros(shape=[self.map_lenght**2,self.map_lenght**2]) + 6 # 初始信息素全部设为6
        j_2 = np.array(self.route_load)
        j_3 = j_2[:,0]*self.map_lenght+j_2[:,1]

        for t in range(len(j_3)-1):
            start_node = j_3[t]
            end_node = j_3[t+1]
            pher_pow = self.init_pher_pow
            distance = 0
            
            for k in range(t+1, len(j_3)):
                next_start_node = j_3[k-1]
                next_end_node = j_3[k]
                distance += abs(next_end_node//self.map_lenght - end_node//self.map_lenght) + abs(next_end_node%self.map_lenght - end_node%self.map_lenght)
                
                if k == len(j_3)-1 or (next_end_node in [start_node, end_node]):
                    # 当前节点是路径终点或者路径上的节点，则将当前节点的信息素设置为6，否则依次递减
                    pher_pow = max(0.5, pher_pow - self.pher_decay_rate * distance)
                    break
                
            pher_data_init[start_node][end_node] *= pher_pow

        self.pher_data = pher_data_init

#############################################################333

        self.evaporate = evaporate_init #信息素挥发系数                                 #########每一步可能的起点(格子)编号总数*下一步可能的编号总数

        self.generation_aver = []  #每代的平均路径(大小)，绘迭代图用
        self.generation_best = []  #每代的最短路径(大小)，绘迭代图用
        self.way_len_best = 999999 
        # self.way_data_best = []     #最短路径对应的节点信息，画路线用  


        self.success_rate_list = []  #记录历代搜索的成功率
        self.cost = []  #记录每代的多因素代价值记录列表
        self.avg_cost_success_rate = []  #记录平均多因素代价记录列表 （记得将avg_cost写成数组）



##################################################
        self.way_zuhe_best = [float('inf')] * 4
        self.way_data_best = [[] for _ in range(4)]
        self.ways_data = [[] for _ in range(4)]    ####用来记录不同权重系数大小的路径数据，画组合路线图用
############################################################



        
    def run(self):

        # 多因素权重系数组合存储在字典中
        weight_combination = {
            0: {'a1': 1, 'a2': 0.853, 'a3': 4.833},
            1: {'a1': 1, 'a2': 0, 'a3': 0},
            2: {'a1': 0, 'a2': 1, 'a3': 0},
            3: {'a1': 0, 'a2': 0, 'a3': 1}
        }
        zuhe=0 #权重系数组合的索引值，其中组合0为聚合多因素

        weights = weight_combination[zuhe]
        a1 = weights['a1']
        a2 = weights['a2']
        a3 = weights['a3']
        #总迭代开始
        for i in range(self.max_iter):      
            success_way_list = []
            print('\n','第',i,'代: ',end = '\n')
            #Step 1:当代若干蚂蚁依次行动     #####这里用的50只蚂蚁
            for j in range(self.ant_num):                                           #######最大行动力是地图宽度的3倍
                ant = Ant(self.node_load,start =self.ant_params['start'],end=self.ant_params['end'], max_step=self.map_lenght*3,pher_imp=self.ant_params['pher_imp'],dis_imp=self.ant_params['dis_imp'])
                ant.run(map_data=self.map_data.copy(),pher_data=self.pher_data)
                if ant.successful == True:  #若成功，则记录路径信息
                    success_way_list.append(ant.record_way)     ###########添加单只蚂蚁成功的路线
            print(' 成功数:',len(success_way_list),end= '  ')###############################################################感觉应该是每只蚂蚁的搜索路径成功的数量？
            #Step 2:计算每条路径对应的长度，经过的节点累积载荷，经过的站点数，后用于信息素的生成量
            way_lenght_list = []
            success_way_node_load_sum = []
            zhandian_count = []

            each_iter_cost =[]

            for j in success_way_list:
                way_lenght_list.append(self.calc_total_lenght(j))         #####统计每条成功路径的长度的list
                ##################################################################################################################
                success_way_node_load_sum.append(self.calc_total_node_load(j))
				##################################################################################################################
                zhandian_count.append(self.calc_total_zhandian(j))
                

            #Step 3:更新信息素浓度
            #  step 3.1: 挥发
            # self.evaporate = self.calc_evaporate(i=i,way_lenght_list=way_lenght_list)#############自适应挥发因子的计算
            # self.pher_data = self.evaporate*self.pher_data     ##########一个挥发系数乘以信息素矩阵（可改为变系数）############
            
            self.evaporate = self.calc_evaporate(i=i,way_lenght_list=way_lenght_list)#############自适应挥发因子的计算
            self.pher_data = (1-self.evaporate)*self.pher_data     ##########一个挥发系数乘以信息素矩阵（可改为变系数）############  1-rou

            #  step 3.2: 叠加新增信息素                       #########################赶紧举例子############################
            for k,j in enumerate(success_way_list):
                j_2 = np.array(j)                    ###############j_2是每一条accord_way组成数组
                j_3 = j_2[:,0]*self.map_lenght+j_2[:,1]         #############这里由于按行展开，所以原始坐标对应onehot编号的关系即为----x*lenght+y
            ########################这里的信息素是表示每一步的路径上的，所以具有方向性
            #########################所以相当于把栅格地图onehot展开，用前后格子的序号来构成二维坐标(表示前进路径方向上的信息素)
                for t in range(len(j_3)-1):
            ######################所以j_3[t]][j_3[t+1]为当前路径上的起点和终点，j_3[t+1]][j_3[t+2]为下一步的起点和终点
                    self.pher_data[j_3[t]][j_3[t+1]] += self.ant_gener_pher/(way_lenght_list[k] + success_way_node_load_sum[k] + zhandian_count[k]) 
                                           ##################################多因素：全路径长度、全路径节点载荷、路径节点个数
                each_iter_cost.append((a1*way_lenght_list[k] + a2*success_way_node_load_sum[k] + a3*zhandian_count[k]))
            print('每代平均载荷值大小：',np.mean(success_way_node_load_sum),'每代平均路径长度：',np.mean(way_lenght_list))############

            #Step 4: 当代的首尾总总结工作
            self.generation_aver.append(np.average(way_lenght_list))
            self.generation_best.append(min(way_lenght_list))

            # if self.way_len_best>min(way_lenght_list):
            #     a_1 = way_lenght_list.index(min(way_lenght_list))
            #     self.way_len_best = way_lenght_list[a_1]
            #     self.way_data_best = copy.deepcopy(success_way_list[a_1])

            zuhecost_list = []
            for x in range(len(way_lenght_list)):
                zuhecost_list.append(a1*way_lenght_list[x] + a2*success_way_node_load_sum[x] + a3*zhandian_count[x])

            if self.way_zuhe_best[zuhe]>min(zuhecost_list):
                a_1 = zuhecost_list.index(min(zuhecost_list))       ####### 找出组合最小损失值（的下标a_1）
                print('index_a_1:',a_1)
                self.way_zuhe_best[zuhe] = zuhecost_list[a_1]
                self.way_data_best[zuhe] = copy.deepcopy(success_way_list[a_1])
                self.way_data_best = self.way_data_best[zuhe]
                self.ways_data[zuhe].append(copy.deepcopy(success_way_list[a_1]))###### 把组合最小损失值对应的success_way_list存储到列表中




            ##############################MACO算法部分的评估指标（每代的成功搜索代价平均值
            self.success_rate_list.append(len(success_way_list)/self.ant_num)   ###每代搜索的成功率
            avg_success_rate = sum(self.success_rate_list) / len(self.success_rate_list)   #####历次迭代的平均搜索成功率
            avg_iter_cost_averge = sum(each_iter_cost) / len(each_iter_cost)   #####每代的成功搜索代价平均值
            self.cost.append(avg_iter_cost_averge)   ###每代的成功搜索代价总值总和/历次迭代
            self.avg_cost_success_rate.append(avg_success_rate)   ###历次迭代的平均搜索成功率



            # Save variables to file
            # with open('LPA+ACO_100ant_duoditu_demo_2.txt', 'a') as f:   ### 从左上角到右下角的一般性地图LPA路径
            with open('LPA+ACO_100ant_duoditu_reverse_demo_1.txt', 'a') as f:   ### 从左下角到右上角的一般性地图LPA路径
                f.write("\n"f"Generation Average route_LEN: {self.generation_aver}\n")
                # f.write(f"Generation Best: {self.generation_best}\n")
                # f.write(f"Best Way Length: {self.way_len_best}\n")
                f.write(f"Best Way Data: {self.way_data_best}\n")
                f.write(f"all_iter_cost: {self.cost}\n")  
                f.write(f"Average cost_success_rate: {self.avg_cost_success_rate}\n")  # per iteration, not per ant.



            #######对信息素矩阵设置上下限
            self.pher_data[self.pher_data>=self.tao_max] = self.tao_max
            self.pher_data[self.pher_data<=self.tao_min] = self.tao_min

            print('平均长度:',np.average(way_lenght_list),'最短:',np.min(way_lenght_list),'最少站点数:',np.min(zhandian_count))         ##############################################也没明白，也是要把数学形式列出来看看
           
            # print('成功路径表',success_way_list)###########
            # print('成功路径长度',way_lenght_list)###########

            print('成功站点个数',zhandian_count)###########

            print('每代成功搜索的代价总值：',each_iter_cost,)############
            print('每代成功搜索的路径长度：',a1*way_lenght_list[k] ,'每代成功搜索的节点载荷：', a2*success_way_node_load_sum[k] ,'每代成功搜索的节点个数：', a3*zhandian_count[k])############
            print('每代成功搜索率：',len(success_way_list)/self.ant_num)############

            print(f'所有迭代的平均成功率为{avg_success_rate}, 每代成功搜索代价的平均值为{avg_iter_cost_averge}')
        #     print('信息素矩阵最大值：',self.pher_data.max())###########
        #     print('信息素矩阵最小值：',self.pher_data.min())###########
        #     print('###############每代信息素挥发因子大小：',self.evaporate)############
        # print('##################################\n',self.pher_data,'\n','####################')
        # print('##################################\n',list(set([element for row in self.pher_data for element in row])),'\n','####################')
        # print('##################################\n',dict(Counter([element for row in self.pher_data for element in row])),'\n','####################')
        
    def calc_total_lenght(self,way):
        lenght = 0
        for j1 in range(len(way)-1):
            a1 = abs(way[j1][0]-way[j1+1][0])+abs(way[j1][1]-way[j1+1][1])      #########曼哈顿距离
            if a1 == 2:
                # lenght += 1.41421    #######这里表示走的斜角距离就是1.414
                lenght += 2
            else:
                lenght += 1         #######反之走的即是直边，长度为1
        return lenght                

    def calc_total_node_load(self,way):
        load = 0
        for j1 in range(len(way)-1):
            load += self.node_load[way[j1][0]][way[j1][1]]
        return load                
    
    def calc_total_zhandian(self,way):
        count = 0
        jiedian_list = []
        with open('./光标站点坐标-多地图_demo.txt','r') as f:
                        a1 = f.readlines()
                        for i in a1:
                             i = i.strip('\n')
                             jiedian_list.append(eval(i))
        for j in way:
            if j in jiedian_list:
                count += 1
        return count
    
    def calc_evaporate(self,i,way_lenght_list,evaporate_min = 0.5,evaporate_max = 0.9):
        if self.evaporate<evaporate_min:
            evaporate = evaporate_min
        elif self.evaporate>evaporate_max:
            evaporate = evaporate_max
        
        else:
            evaporate = (self.max_iter-i) / self.max_iter
            evaporate = evaporate*self.pher_data.max()/self.tao_max
            evaporate = evaporate * max(way_lenght_list)/(3*self.map_lenght)
            evaporate = evaporate + 0.3

        return evaporate
