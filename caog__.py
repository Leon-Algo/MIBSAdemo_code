matrix = [[0,-1,4,3,-1],
[5,0,-1,1,2],
[-1,-1,0,6,9],
[7,-1,-1,0,-1],
[-1,-1,-1,4,0]]
for row in matrix:
    print(row)

def floyd(W):
    # 首先获取节点数
    node_number = len(W)

    # 初始化路由矩阵
    R = [[0 for i in range(node_number)] for j in range(node_number)]
    for i in range(node_number):
        for j in range(node_number):
            if W[i][j] > 0:
                R[i][j] = j+1
            else:
                R[i][j] = 0
    # # 查看初始化的路由矩阵
    # for row in R:
    #     print(row)

    # 循环求W_n和R_n
    for k in range(node_number):
        for i in range(node_number):
            for j in range(node_number):
                if W[i][k] > 0 and W[k][j] > 0 and (W[i][k] + W[k][j] < W[i][j] or W[i][j] == -1):
                    W[i][j] = W[i][k] + W[k][j]
                    R[i][j] = k+1
        print("第%d次循环:" % (k+1))
        print("距离矩阵:")
        for row in W:
            print(row)
        print("路由矩阵:")
        for row in R:
            print(row)


floyd(matrix)