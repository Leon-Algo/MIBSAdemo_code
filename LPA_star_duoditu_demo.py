
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
 
 
class PriorityQueue:
    def __init__(self):
        self.queue = OrderedDict()
 
    def insert(self, vertex, key):
        self.queue[vertex] = key
        self.queue = OrderedDict(sorted(self.queue.items(), key=lambda kv: (kv[1][0], kv[1][1])))
 
    def popvertex(self):
        first_item = self.queue.popitem(last=False)
        return first_item[0]
 
    def topkey(self):
        if self.queue:
            item = next(iter(self.queue.items()))
            return item[1]
        else:
            return np.inf, np.inf
 
    def remove(self, k):
        if k in self.queue:
            print('(%s, %s) to remove from priority queue' % (k[0], k[1]))
            del self.queue[k]
 
 
g = dict()
rhs = dict()
h = dict()
U = PriorityQueue()
 
 
# # Eight Directions
# neighbor_direction = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
# def heuristic(a, b):
#     return max(abs(a[0]-b[0]), abs(a[1]-b[1]))
 
 
# Four Directions
neighbor_direction = [(0, 1), (0, -1), (1, 0), (-1, 0)]
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])
 

def get_neighbor(current):
    neighbors = list()
    for i, j in neighbor_direction:
        neighbor = current[0] + i, current[1] + j
        if 0 <= neighbor[0] < S.shape[0]:
            if 0 <= neighbor[1] < S.shape[1]:
                if S[neighbor[0]][neighbor[1]] == 1:
                    # print('neighbor %s hit wall' % str(neighbor))
                    continue
            else:
                # array bound y walls
                # print('neighbor %s hit y walls' % str(neighbor))
                continue
        else:
            # array bound x walls
            # print('neighbor %s hit x walls' % str(neighbor))
            continue
 
        neighbors.append(neighbor)
 
    return neighbors
 
 
def update_vertex(s):
    print('---update_vertex for (%s, %s)' % (s[0], s[1]))
 
    if s != start:
        neighbors = get_neighbor(s)
        print('(%s, %s) get neighbors:' % (s[0], s[1]))
        print(neighbors)
        neighbor_g_c_s = list()
        for nb in neighbors:
            g_c = g[nb] + heuristic(nb, s)
            neighbor_g_c_s.append(g_c)
        rhs[s] = min(neighbor_g_c_s)
        print('neighbor g_c: %s' % neighbor_g_c_s)
        print(rhs[s])
 
    U.remove(s)
 
    if g[s] != rhs[s]:
        U.insert(s, calculate_key(s))
 
 
def initialize():
    print('start initialize...')
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[(i, j)] != 1:
                g[(i, j)] = np.inf
                rhs[(i, j)] = np.inf
 
    rhs[start] = 0
 
    init_h()
    U.insert(start, (h[start], 0))
    print('Initial U:')
    print(U.queue)
    print('finish initialize...')
 
 
def init_h():
    S_shape = S.shape
    for i in range(S_shape[0]):
        for j in range(S_shape[1]):
            if S[(i, j)] != 1:
                node = (i, j)
                h_calc = heuristic(node, goal)
                h[node] = h_calc                   
 
##原始版k值计算
# def calculate_key(s):
#     return min(g[s], rhs[s]) + h[s], min(g[s], rhs[s])
 
#加p和w的k值计算 
def calculate_key(s):
    point1 = np.array(start) 
    point2 = np.array(goal) 
    toal_lenght = np.linalg.norm(point1 - point2)
    w = (h[s]/toal_lenght + 1)              #################一个变化的动态权重
    # p = 1/((S.shape[0])**2)                ########搜索偏置p----减少搜索K值相同时，不同分支要搜索的点---只搜索一个分支
 #########################################################   
    
    dx1 = s[0] - goal[0]
    dy1 = s[1] - goal[1]
    dx2 = start[0] - goal[0]
    dy2 = start[1] - goal[1]
    cross = abs(dx1 * dy2 - dx2 * dy1)
    p= cross * 0.001
    
    # return min(g[s], rhs[s]) + h[s]*(w+p), min(g[s], rhs[s])             #######乘一个权重w，再加一个偏置p
    return min(g[s], rhs[s]) + h[s]*w+p, min(g[s], rhs[s])  
 
 
def computer_shortest_path():
    computer_iteration = 1
    while U.topkey() < calculate_key(goal) or rhs[goal] != g[goal]:
        print('----------------Iteration #%d----------------' % computer_iteration)
        u = U.popvertex()
        print('top_vertex (%s, %s)' % (u[0], u[1]))
        if g[u] > rhs[u]:
            print('overconsistent as (%s, %s) g_value %s > rhs_value %s' % (u[0], u[1], g[u], rhs[u]))
            g[u] = rhs[u]
            print('set (%s, %s) g_value %s same as rhs_value %s' % (u[0], u[1], g[u], rhs[u]))
            neighbors = get_neighbor(u)
            for nb in neighbors:
                update_vertex(nb)
        else:
            g[u] = np.inf
            neighbors = get_neighbor(u)
            for nb in neighbors:
                update_vertex(nb)
 
            update_vertex(u)
 
        print(U.queue)
        computer_iteration += 1
 
    print('---exit computer shortest path as reach terminate condition---')
 
 
def get_path_node():
    path_node = list()
    path_node.append(goal)
 
    current = goal
    while current != start:
        nbs = get_neighbor(current)
        nbs_rhs_dict = OrderedDict()
 
        for nb in nbs:
            nbs_rhs_dict[nb] = rhs[nb]
 
        sorted_nbs_rhs_dict = OrderedDict(sorted(nbs_rhs_dict.items(), key=lambda item: item[1]))
        trace_back_node_rhs = sorted_nbs_rhs_dict.popitem(last=False)
        trace_back_node = trace_back_node_rhs[0]
        path_node.append(trace_back_node)
 
        current = trace_back_node
 
    return path_node
 
 
if __name__ == '__main__':
    #通用地图
    S = np.array([
        [0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,],
        [0,1,1,0,1,1,1,1,1,1,0,1,0,1,1,1,],
        [0,1,1,0,1,1,1,1,1,1,0,0,0,1,1,1,],
        [0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,],
        [1,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,],
        [1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,],
        [1,1,0,1,1,0,0,0,0,1,1,1,0,0,1,0,],
        [1,1,0,1,0,0,1,1,0,0,0,0,0,0,1,0,],
        [1,1,0,0,0,0,1,1,0,1,1,0,1,0,1,0,],
        [0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,],
        [0,1,0,0,0,1,1,1,0,1,1,1,1,0,0,1,],
        [0,1,1,1,0,1,1,0,0,1,1,1,1,0,0,1,],
        [0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,],
        [0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,0,],
        [0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,],
        [0,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,],
    ])   


# # 第一种从左上至右下的实验搜索路径
#     start = (0, 0)
#     goal = (14,15)
# 第二种从左下至右上的实验搜索路径
    start = (14, 0)
    goal = (5,15)
	
 
    # # # test if there is no route
    # # S[(4, 1)] = 1
    # # S[(5, 1)] = 1
 
    # below start run
    initialize()
    computer_shortest_path()
 
    # # --------------------------------------------
    # # this section to test if the environment change
    # print()
    # print()
    # print('--------after update:')
    #
    # # change_point = (3, 1)
    # change_point = (7, 12)
    # S[change_point] = 1
    # g[change_point] = np.inf
    # update_vertex(change_point)
    #
    # print(U.queue)
    # computer_shortest_path()
    # # ----------------------------------------------
 
    if rhs[goal] == np.inf:
        path_exist = False
    else:
        path_exist = True
 
    if path_exist:
        print()
        route = get_path_node()
        print('route:')
        print(route)
 
    plot_map_and_path = True
 
    # if plot_map_and_path:
    #     # plot map and path
    #     fig, ax = plt.subplots(figsize=(20, 20))
    #     ax.imshow(S, cmap=plt.cm.Greys)   #matplotlib内置的颜色图Greys
 
    #     ax.scatter(start[1], start[0], marker="o", color="red", s=200)
    #     ax.scatter(goal[1], goal[0], marker="*", color="blue", s=200)
 
    #     if path_exist:
    #         # extract x and y coordinates from route list
    #         x_coords = []
    #         y_coords = []

    #         for k in (range(0, len(route))):
    #             x = route[k][0]
    #             y = route[k][1]
    #             x_coords.append(x)
    #             y_coords.append(y)
 
    #         ax.plot(y_coords, x_coords, color="black")
 
    #     ax.xaxis.set_ticks(np.arange(0, S.shape[1], 1))
    #     ax.yaxis.set_ticks(np.arange(0, S.shape[0], 1))
    #     ax.xaxis.tick_top()
    #     plt.grid()
    #     plt.show()

    if plot_map_and_path:
        # plot map and path
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(S, cmap=plt.cm.Greys)   #matplotlib内置的颜色图Greys

        if path_exist:
            # extract x and y coordinates from route list
            x_coords = []
            y_coords = []

            for k in range(len(route)):
                x = route[k][0]
                y = route[k][1]
                x_coords.append(x)
                y_coords.append(y)

            ax.plot(y_coords, x_coords, color="black")

        ax.set_xticks(np.arange(0, S.shape[1], 1))
        ax.set_yticks(np.arange(0, S.shape[0], 1))
        ax.xaxis.tick_top()
        plt.grid()
        plt.show()


    # if plot_map_and_path:
    #     # plot map and path
    #     fig, ax = plt.subplots(figsize=(20, 20))
    #     ax.imshow(S, cmap='binary', extent=[0, S.shape[1], S.shape[0], 0])

    #     if path_exist:
    #         # extract x and y coordinates from route list
    #         x_coords = []
    #         y_coords = []

    #         for k in range(len(route)):
    #             x = route[k][1]
    #             y = route[k][0]
    #             x_coords.append(x)
    #             y_coords.append(y)

    #         ax.plot(x_coords, y_coords, color="black")

    #     ax.set_xticks(np.arange(0, S.shape[1], 1))
    #     ax.set_yticks(np.arange(0, S.shape[0], 1))
    #     ax.xaxis.tick_top()
    #     plt.grid()
    #     plt.show()


        # 保存搜索路径列表
        file = open('./route_list_4diriction_duoditu_reverse.txt', 'w')
        for fp in route[::-1]:
            file.write(str(fp))
            file.write('\n')
        file.close()
        print('保存路径成功！')
        

