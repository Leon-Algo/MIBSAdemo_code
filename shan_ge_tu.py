import numpy as np
import math
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

infinity = float('inf')
# 栅格界面场景定义
rows = 30
cols = 50
# 定义栅格地图全域，并初始化空白区域
field = np.ones(rows * cols)
# 起始点和目标点
start = 2
goal = rows * cols - 2
# 障碍物区域
obsRate = 0.3  # 障碍物出现几率
obsNum = math.floor(obsRate * rows * cols)
obsIndex = random.sample(range(0, rows * cols), obsNum)
field[obsIndex] = 2
field[start] = 4
field[goal] = 5