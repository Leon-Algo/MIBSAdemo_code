import numpy as np
import pandas as pd
import copy

################################################## 1 蚁群算法路径规划 ###########################################


# Ant只管通过地图数据以及信息素数据，输出一条路径。其他的你不用管。
class Ant():
    def __init__(self,start,end,max_step,pher_imp,dis_imp) -> None:
        self.max_step = max_step    # 蚂蚁最大行动力
        self.pher_imp = pher_imp    # 信息素重要性系数
        self.dis_imp = dis_imp      # 距离重要性系数
        self.start = start          # 蚂蚁初始位置[y,x] = [0,0],考虑到列表索引的特殊性，先定y，后定x
        self.destination = end  # 默认的终点节点(在run方法中会重新定义该值)
        self.successful = True      #标志蚂蚁是否成功抵达终点
        self.record_way = [start]   #路径节点信息记录
        

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
        node_be_selected = [[y_1-1,x_1-1],[y_1-1,x_1],[y_1-1,x_1+1],     #上一层
                            [y_1,x_1-1],              [y_1,x_1+1],       #同层
                            [y_1+1,x_1-1],[y_1+1,x_1],[y_1+1,x_1+1],     #下一层
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
        dis_1 = []    # 距离启发因子
        for i in node_be_selected_1:
            dis_1.append(((self.destination[0]-i[0])**2+(self.destination[1]-i[1])**2)**0.5)
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
    def __init__(self,map_data,start = [0,0],end = [19,19],max_iter = 100,ant_num = 50,pher_imp = 1,dis_imp = 10,evaporate = 0.7,pher_init = 8) -> None:
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
        self.ant_gener_pher = 1    #每只蚂蚁携带的最大信息素总量
        self.pher_init = pher_init #初始信息素浓度
        self.ant_params = {        #生成蚂蚁时所需的参数
            'dis_imp':dis_imp,
            'pher_imp': pher_imp,
            'start'   : start,
            'end'     : end
        }
        self.map_data = map_data.copy()        #地图数据
        self.map_lenght = self.map_data.shape[0]  #地图尺寸,用来标定蚂蚁的最大体力           ########地图尺寸的平方--因为路径上的信息素具有方向性
        self.pher_data = pher_init*np.ones(shape=[self.map_lenght*self.map_lenght, #################################################
                                            self.map_lenght*self.map_lenght])    ##################信息素矩阵################（方阵）！！
        self.evaporate = evaporate #信息素挥发系数                                 #########每一步可能的起点(格子)编号总数*下一步可能的编号总数
        self.generation_aver = []  #每代的平均路径(大小)，绘迭代图用
        self.generation_best = []  #每代的最短路径(大小)，绘迭代图用
        self.way_len_best = 999999 
        self.way_data_best = []     #最短路径对应的节点信息，画路线用  


        
    def run(self):
        #总迭代开始
        for i in range(self.max_iter):      
            success_way_list = []
            print('第',i,'代: ',end = '')
            #Step 1:当代若干蚂蚁依次行动     #####这里用的50只蚂蚁
            for j in range(self.ant_num):                                           #######最大行动力是地图宽度的3倍
                ant = Ant(start =self.ant_params['start'],end=self.ant_params['end'], max_step=self.map_lenght*3,pher_imp=self.ant_params['pher_imp'],dis_imp=self.ant_params['dis_imp'])
                ant.run(map_data=self.map_data.copy(),pher_data=self.pher_data)
                if ant.successful == True:  #若成功，则记录路径信息
                    success_way_list.append(ant.record_way)     ###########添加单只蚂蚁成功的路线
            print(' 成功数:',len(success_way_list),end= '  ')###############################################################感觉应该是每只蚂蚁的搜索路径成功的数量？
            #Step 2:计算每条路径对应的长度，后用于信息素的生成量
            way_lenght_list = []
            for j in success_way_list:
                way_lenght_list.append(self.calc_total_lenght(j))         #####统计每条成功路径的长度的list
            #Step 3:更新信息素浓度
            #  step 3.1: 挥发
            self.pher_data = self.evaporate*self.pher_data     ##########一个挥发系数乘以信息素矩阵（可改为变系数）############
            #  step 3.2: 叠加新增信息素                       #########################赶紧举例子############################
            for k,j in enumerate(success_way_list):
                j_2 = np.array(j)                    ###############j_2是每一条accord_way组成数组
                j_3 = j_2[:,0]*self.map_lenght+j_2[:,1]         #############这里由于按行展开，所以原始坐标对应onehot编号的关系即为----x*lenght+y
            ########################这里的信息素是表示每一步的路径上的，所以具有方向性
            #########################所以相当于把栅格地图onehot展开，用前后格子的序号来构成二维坐标(表示前进路径方向上的信息素)
                for t in range(len(j_3)-1):
            ######################所以j_3[t]][j_3[t+1]为当前路径上的起点和终点，j_3[t+1]][j_3[t+2]为下一步的起点和终点
                    self.pher_data[j_3[t]][j_3[t+1]] += self.ant_gener_pher/way_lenght_list[k]          ##################################这要在找点数据例子看看，把数学形式给写出来
            #Step 4: 当代的首尾总总结工作
            self.generation_aver.append(np.average(way_lenght_list))
            self.generation_best.append(min(way_lenght_list))
            if self.way_len_best>min(way_lenght_list):
                a_1 = way_lenght_list.index(min(way_lenght_list))
                self.way_len_best = way_lenght_list[a_1]
                self.way_data_best = copy.deepcopy(success_way_list[a_1])
            print('平均长度:',np.average(way_lenght_list),'最短:',np.min(way_lenght_list))         ##############################################也没明白，也是要把数学形式列出来看看
           
            print('成功路径表',success_way_list)###########
            print('成功路径长度',way_lenght_list)###########

    
    def calc_total_lenght(self,way):
        lenght = 0
        for j1 in range(len(way)-1):
            a1 = abs(way[j1][0]-way[j1+1][0])+abs(way[j1][1]-way[j1+1][1])      #########切比雪夫距离
            if a1 == 2:
                lenght += 1.41421    #######这里表示走的斜角距离就是1.414
            else:
                lenght += 1         #######反之走的即是直边，长度为1
        return lenght                


          
# class A_Star_path():
#     def __init__(self,map_data,start = np.array([0,0]),end = np.array([0,0])) -> None:
#         print('A* Task:',start,end)
#         self.map_data = np.array(map_data)
#         self.size_map = self.map_data.shape[0]
#         self.start = start
#         self.end   =  end
#         self.open_list  = pd.DataFrame([[start[0],start[1],0,abs(self.start[0]-self.end[0])+abs(self.start[1]-self.end[1]),abs(self.start[0]-self.end[0])+abs(self.start[1]-self.end[1]),-1,-1]],columns=['pos_y','pos_x','g','h','f','parent_y','parent_x'])
#         self.close_list = pd.DataFrame([[999,999,0,0,0,0,0]],columns=['pos_y','pos_x','g','h','f','parent_y','parent_x'])
#         self.best_way_len = 0
        

#     def run(self):
#         running = True
#         while running:
#             #step 1: select an element whose f is smallest(if there are multiple elements with the same f value, we will select the one with the lowest s value.)
#             smal_index = self.open_list.loc[:,'f'].argmin()
#             #step 2: find all the neighbor grid of it
#             neighbour_allowed = self._find_neighbour(self.open_list.iloc[smal_index].values)
#             #step 3: update open_list by neighbour_allowed
#             have_find_term = self._update_open_list(neighbour_allowed,self.open_list.iloc[smal_index].values)
#             if have_find_term == True:
#                 break
#             #step 4: delete it in the open_list and copy to the close_list
#             self.close_list.loc[self.close_list.shape[0]] = self.open_list.iloc[smal_index].copy()
#             self.open_list.drop(smal_index,inplace=True)
#             self.open_list.index = range(self.open_list.shape[0])
#         self._calc_way()
#         #!print('最优路径为:')
#         #!(self.best_way_data)
        


              

#     def _find_neighbour(self,who):
#         '''
#             Function:
#             ---------
#             This Func can find all neighbor grid of the who, and return them.

#             Params:
#             -------
#             who : 1*d-list--> fe:[[0,0],0,19,19,-1]

#             Return:
#             ---------
#             neighbour_allowed : 1*d-list--> all the neighbour can allowed.
#         '''
#         y = int(who[0])
#         x = int(who[1])
#         neighbour_1 = [
#             [y-1,x-1],[y-1,x],[y-1,x+1],
#             [y  ,x-1],        [y  ,x+1],
#             [y+1,x-1],[y+1,x],[y+1,x+1],
#         ]
#         neighbour_allowed = []
#         for i in neighbour_1:
#             if (0<=i[0]<=self.size_map-1)and(0<=i[1]<=self.size_map-1): #grid within the scope of the map.
#                 if self.map_data[i[0],i[1]] == 0:
#                     neighbour_allowed.append(i)
#         return neighbour_allowed

#     def _update_open_list(self,neighbour_allowed,who):
#         have_find_term = False           # maiks whether the terminal was found
#         for i in neighbour_allowed:
#             # case 1: if i in close_list, discard it. 
            
#             if (i == self.close_list.loc[:,['pos_y','pos_x']].values).all(axis = 1).any():
#                 continue
#             # we must calculate the data first as it will be used in the both following case. 
#             dict_1 = {
#                 'pos_y' : i[0],
#                 'pos_x' : i[1],
#                 'g'   : who[2]+(1.414 if (abs(i[0]-who[0])+abs(i[1]-who[1]))==2 else 1),
#                 'h'   : abs(i[0]-self.end[0])+abs(i[1]-self.end[1]),
#                 'f'   : 0,
#                 'parent_y': who[0],
#                 'parent_x': who[1]
#             }
#             dict_1['f']  = dict_1['g']+dict_1['h']
#             # case 2: provided i in open_list, check whether i's f is greater 
#             #              than b's, if True, update the data of i in the open_list;
#             #              otherwise, discard new it. 
#             r = (i == self.open_list.loc[:,['pos_y','pos_x']].values).all(axis = 1)
#             if r.any():
#                 index = np.where(r==True)[0]
#                 if self.open_list.loc[index,'f'].values>dict_1['f']:
#                     self.open_list.loc[index,'g'] = dict_1['g']
#                     self.open_list.loc[index,'h'] = dict_1['h']
#                     self.open_list.loc[index,'f'] = dict_1['f']
#                     self.open_list.loc[index,'parent_y'] = dict_1['parent_y']
#                     self.open_list.loc[index,'parent_x'] = dict_1['parent_x']
                    
#                 else:
#                     pass
#             # case 3: add the i in the open_list.(when i is terminal,the return_param will be True)
#             else:
#                 self.open_list.loc[self.open_list.shape[0]] = dict_1
#             if (i == self.end).all():
#                 have_find_term = True
#         return have_find_term

#     def _calc_way(self):
#         '''
#             Function:
#             ---------
#                 find the best way from open_list and close_list, and the result will be saved in self.best_way_data.
#         '''
#         #step 1: Ectract the position and parent of each element in open_list and close_list.
#         dict_1 = {

#         }
#         for k in range(self.open_list.shape[0]):
#             i = self.open_list.iloc[k]
#             dict_1[str(int(i.values[0]))+','+str(int(i.values[1]))] = str(int(i.values[-2]))+','+str(int(i.values[-1]))
#         for k in range(self.close_list.shape[0]):
#             i = self.close_list.iloc[k]
#             dict_1[str(int(i.values[0]))+','+str(int(i.values[1]))] = str(int(i.values[-2]))+','+str(int(i.values[-1]))
#         # step 2: find the best way between the end with start from dict_1.
#         y,x = self.end
#         self.best_way_data = [[y,x]]
#         while True:
#             a_1 = dict_1[str(y)+','+str(x)].split(',')
#             y,x = eval(a_1[0]),eval(a_1[1])
#             b_1 = self.best_way_data[-1]
#             self.best_way_len += ((y-b_1[0])**2+(x-b_1[1])**2)**0.5
#             self.best_way_data.append([y,x])
#             if y==self.start[0] and x == self.start[1]:
#                 break
#         self.best_way_data.reverse()
            





        
        



        
        

        



