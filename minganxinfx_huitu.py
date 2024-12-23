# # 


# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# x = np.arange(5)
# x_labels = ['Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5']
# y1 = np.array([39.32, 1.98, 0.50, 0.33, 34.56])
# y2 = np.array([11.46, 9.06, 12.41, 6.58, 5.06])
# y3 = np.array([6.65, 6.04, 4.53, 6.15, 6.20])
# y4 = np.array([8.33, 5.40, 10.67, 12.10, 5.29])
# y5 = np.array([5.44, 6.26, 4.85, 5.50, 12.72])

# # 绘图
# fig, ax1 = plt.subplots(figsize=(8, 6))

# # 设置x轴标签和刻度
# ax1.set_xticks(x)
# ax1.set_xticklabels(x_labels, fontsize=12)

# # 折线图绘制
# lns1 = ax1.plot(x, y1, label='Information Pheromone', marker='o', linewidth=2, color='tab:blue')
# lns2 = ax1.plot(x, y2, label='Heuristic Value', marker='s', linewidth=2, color='tab:orange')
# lns3 = ax1.plot(x, y3, label='Initial Pheromone Concentration', marker='^', linewidth=2, color='tab:green')
# lns4 = ax1.plot(x, y4, label='Evaporation Coefficient', marker='*', linewidth=2, color='tab:red')
# lns5 = ax1.plot(x, y5, label='Max Pheromone Carrying Capacity', marker='d', linewidth=2, color='tab:purple')

# # 设置y轴1上限
# ax1.set_ylim([0, 50])

# # 添加标题和y轴1标签
# ax1.set_title('Sensitivity Analysis of Loss Function', fontsize=18)
# ax1.set_ylabel('Loss Evaluation', fontsize=14)

# # 添加网格线
# ax1.grid(axis='y', linestyle='--', alpha=0.7)

# # 右侧y轴绘制条形图
# ax2 = ax1.twinx()

# # 调整条形图位置和颜色
# bar_width = 0.15
# ax2.bar(x - bar_width*2, y1, color='tab:blue', width=bar_width, label='Information Pheromone')
# ax2.bar(x - bar_width, y2, color='tab:orange', width=bar_width, label='Heuristic Value')
# ax2.bar(x, y3, color='tab:green', width=bar_width, label='Initial Pheromone Concentration')
# ax2.bar(x + bar_width, y4, color='tab:red', width=bar_width, label='Evaporation Coefficient')
# ax2.bar(x + bar_width*2, y5, color='tab:purple', width=bar_width, label='Max Pheromone Carrying Capacity')

# # 设置y轴2上限
# ax2.set_ylim([0, 50])

# # 添加y轴2的标签
# ax2.set_ylabel('Scale', fontsize=14)

# # 调整图例样式和位置
# lns = lns1+lns2+lns3+lns4+lns5
# labels = [l.get_label() for l in lns]
# leg1 = fig.legend(lns, labels, bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
# leg1.set_title('Line Plot', prop={'size':16, 'weight':'bold'})
# for line in leg1.get_lines():
#     line.set_linewidth(3.0)
# plt.setp(leg1.get_title(), fontsize=16, fontweight='bold')
# plt.subplots_adjust(right=0.85, top=0.85)

# bar_labels = ['Information Pheromone', 'Heuristic Value', 'Initial Pheromone Concentration', 'Evaporation Coefficient', 'Max Pheromone Carrying Capacity']
# leg2 = fig.legend(bar_labels, bbox_to_anchor=(1.05, 0.48), loc='upper left', frameon=False)
# leg2.set_title('Bar Plot', prop={'size':16, 'weight':'bold'})
# for line in leg2.get_lines():
#     line.set_visible(False)
# plt.setp(leg2.get_title(), fontsize=16, fontweight='bold')

# # # 图例1：将第一个子图的图例放在外部展示，并调整格式和位置
# # handles1, labels1 = ax1.get_legend_handles_labels()
# # leg1 = fig.legend(handles1, labels1, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.15))
# # for legobj in leg1.legendHandles:
# #     legobj.set_alpha(1)
# # plt.subplots_adjust(top=0.85)

# # # 图例2：将第二个子图的图例放在外部展示，并调整格式和位置
# # handles2, labels2 = ax2.get_legend_handles_labels()
# # leg2 = fig.legend(handles2, labels2, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.05))
# # for legobj in leg2.legendHandles:
# #     legobj.set_alpha(1)

# # 调整布局
# fig.tight_layout()

# # 显示图形
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 数据
x = np.arange(5)
x_labels = ['Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5']
y1 = np.array([39.32, 1.98, 0.50, 0.33, 34.56])
y2 = np.array([11.46, 9.06, 12.41, 6.58, 5.06])
y3 = np.array([6.65, 6.04, 4.53, 6.15, 6.20])
y4 = np.array([8.33, 5.40, 10.67, 12.10, 5.29])
y5 = np.array([5.44, 6.26, 4.85, 5.50, 12.72])

# 绘图
fig, ax1 = plt.subplots(figsize=(8, 6))

# 设置x轴标签和刻度
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=12)

# 折线图绘制
lns1 = ax1.plot(x, y1, label='Information Pheromone', marker='o', linewidth=2, color='tab:blue')
lns2 = ax1.plot(x, y2, label='Heuristic Value', marker='s', linewidth=2, color='tab:orange')
lns3 = ax1.plot(x, y3, label='Initial Pheromone Concentration', marker='^', linewidth=2, color='tab:green')
lns4 = ax1.plot(x, y4, label='Evaporation Coefficient', marker='*', linewidth=2, color='tab:red')
lns5 = ax1.plot(x, y5, label='Max Pheromone Carrying Capacity', marker='d', linewidth=2, color='tab:purple')


    
# 设置y轴1上限
ax1.set_ylim([-100,80])

# 添加标题和y轴1标签
ax1.set_title('Sensitivity Analysis of Loss Function', fontsize=18)
ax1.set_ylabel('Loss Evaluation', fontsize=14)

# 添加网格线
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 右侧y轴绘制条形图
ax2 = ax1.twinx()

# 调整条形图位置和颜色
bar_width = 0.15
ax2.bar(x - bar_width*2, y1, color='tab:blue', width=bar_width, label='Information Pheromone')

ax2.bar(x - bar_width, y2, color='tab:orange', width=bar_width, label='Heuristic Value')

ax2.bar(x, y3, color='tab:green', width=bar_width, label='Initial Pheromone Concentration')

ax2.bar(x + bar_width, y4, color='tab:red', width=bar_width, label='Evaporation Coefficient')

ax2.bar(x + bar_width*2, y5, color='tab:purple', width=bar_width, label='Max Pheromone Carrying Capacity')





# 设置y轴2上限
ax2.set_ylim([0, 100])

# 添加y轴2的标签
ax2.set_ylabel('Scale', fontsize=14)

# 调整图例样式和位置
lns = lns1+lns2+lns3+lns4+lns5
labels = [l.get_label() for l in lns]
leg1 = fig.legend(lns, labels, bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
leg1.set_title('Line Plot', prop={'size':16, 'weight':'bold'})
for line in leg1.get_lines():
    line.set_linewidth(3.0)
plt.setp(leg1.get_title(), fontsize=16, fontweight='bold')
plt.subplots_adjust(right=0.85, top=0.85)

bar_labels = ['Information Pheromone', 'Heuristic Value', 'Initial Pheromone Concentration', 'Evaporation Coefficient', 'Max Pheromone Carrying Capacity']
leg2 = fig.legend(bar_labels, bbox_to_anchor=(1.05, 0.48), loc='upper left', frameon=False)
leg2.set_title('Bar Plot', prop={'size':16, 'weight':'bold'})
for line in leg2.get_lines():
    line.set_visible(False)
plt.setp(leg2.get_title(), fontsize=16, fontweight='bold')

# 调整布局
fig.tight_layout()

# 显示图形
plt.show()