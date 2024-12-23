############################################################################################################################################################################
###输入数据格式转换
import pandas as pd

# 定义一个函数来处理单个数据集
def process_data(file_path):
    # 加载数据
    data = pd.read_csv(file_path)
    
    # 如果存在 "Unnamed: 0" 列，则删除
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    
    # 将 "timestamp" 列转换为 datetime 类型
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # 删除 "parameter" 列
    data = data.drop(columns=["parameter"])
    
    # 如果存在重复的时间戳和特征名称，通过取平均值来处理
    data_grouped = data.groupby(["timestamp", "name"]).mean().reset_index()
    
    # 进行透视操作
    pivoted_data = data_grouped.pivot(index='timestamp', columns='name', values='value')

    # 删除指定的列
    columns_to_remove = ['energy_consumption', 'energy_consumption_kwh', 'time_of_usage']
    pivoted_data.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    # 对所有值进行绝对化处理
    features = ['ambient_temp', 'dc_bus_volt', 'motor_current', 'motor_power', 'motor_speed', 'motor_torque', 'igbt_junction_temp']
    pivoted_data[features] = pivoted_data[features].abs()
    
    # 根据时间戳对数据进行一分钟重采样，并取平均值
    resampled_data = pivoted_data.resample('1T').mean().reset_index()
    
    return resampled_data

############################################################################################################################################################################
### 请最新待预处理的数据文件名修改到下方
# 文件路径列表
file_paths = [f"./drive_data{i}.csv" for i in range(1, 7)]

# 使用上述函数处理每个数据集，并将处理后的数据存储在一个列表中
processed_data_list = [process_data(file) for file in file_paths]

# 拼接所有处理后的数据集
combined_data = pd.concat(processed_data_list, ignore_index=True)

# 删除 'igbt_junction_temp' 缺失的行数据
combined_data_cleaned = combined_data.dropna(subset=['igbt_junction_temp'])

# 剔除每行数据缺失值个数在两个以上的数据
final_data = combined_data_cleaned[combined_data_cleaned.isnull().sum(axis=1) < 1]

# 将未命名列删除
if "Unnamed: 0" in final_data.columns:
	final_data = final_data.drop(columns=["Unnamed: 0"])
    
# 将目标变量 igbt_junction_temp 移动到最后一列
target_column = final_data['igbt_junction_temp']
final_data = final_data.drop(columns=['igbt_junction_temp'])
final_data['igbt_junction_temp'] = target_column


# 定义保存的文件名及路径
output_file_path = './processed_drive_data_na0.csv'
# 保存合并后的DataFrame到Excel文件
final_data.to_csv(output_file_path)





############################################################################################################################################################################
###数据分析与预处理
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
# from keras.models import Sequential, Model
# from keras.layers import Dense, LSTM, Dropout, concatenate, Input
# from keras.regularizers import l2
# from keras.utils import plot_model
from matplotlib import pyplot as plt
import seaborn as sns
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.layers import Conv1D, Bidirectional, GRU, BatchNormalization
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 将字体设置为 Arial Unicode MS

# 加载数据
data = final_data.copy()

# 转换timestamp列为datetime类型
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

# 删除Unnamed: 0列
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# 基于IQR方法计算异常值的范围
Q1 = data.iloc[:, 1:].quantile(0.25)
Q3 = data.iloc[:, 1:].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值的范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 剔除异常值
data = data[((data.iloc[:, 1:] >= lower_bound) & (data.iloc[:, 1:] <= upper_bound)).all(axis=1)]

# # 显示更新后的数据的维度
# data.shape
# #剔除时间戳列
# data = data.drop(columns='timestamp')
# Get the basic statistics of the data
statistics = data.describe()

# Set the style of the plots
sns.set_style("whitegrid")


fig_d, axs = plt.subplots(3, 3, figsize=(20, 15))

# Remove the last plot as there are only 8 features
fig_d.delaxes(axs[2,2])

for i, column in enumerate(data.columns[1:]):
    axs[i//3, i%3].hist(data[column], bins=30, color='skyblue', edgecolor='black')
    axs[i//3, i%3].set_title(f'Distribution of {column}')

plt.tight_layout()
plt.show()
fig_d.savefig('ABB_原始数据特征分布.png')
print(statistics)


# 接下来，我将创建一个相关性热图，以便我们更好地理解各变量之间的相关性。这将帮助我们理解哪些特征可能会对我们的目标变量`igbt_junction_temp_ave(5.11)`产生影响。
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

corr = data.corr()
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,  cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Heatmap', fontsize=15)
plt.show()
plt.savefig('ABB_原始数据相关性热图.png')

### 将电压的双模态分布,通过kmeans聚类自适应的划分为两个不同工况下的特征
from sklearn.cluster import KMeans
import numpy as np

# 提取dc_bus_volt(1.11)的非空值
volt_data = data['dc_bus_volt'].dropna().values.reshape(-1, 1)

# 使用K-means聚类分析找到两个集群
kmeans = KMeans(n_clusters=2, random_state=0).fit(volt_data)
centers = kmeans.cluster_centers_

# 确定两个区间的界限
lower_bound_1 = centers.min() - (centers.max() - centers.min()) / 2
upper_bound_1 = centers.min() + (centers.max() - centers.min()) / 2
lower_bound_2 = upper_bound_1
upper_bound_2 = centers.max() + (centers.max() - centers.min()) / 2

print(f'工况1的电压区间{lower_bound_1}:{upper_bound_1}')
print(f'工况2的电压区间{lower_bound_2}:{upper_bound_2}')


# 创建新特征
data['dc_bus_volt_cluster_1'] = data['dc_bus_volt'].apply(lambda x: x if lower_bound_1 <= x <= upper_bound_1 else 0)
data['dc_bus_volt_cluster_2'] = data['dc_bus_volt'].apply(lambda x: x if lower_bound_2 <= x <= upper_bound_2 else 0)

# data.drop('dc_bus_volt(1.11)', axis=1)

print(f'将电压分为两种工况后数据shape大小{data.shape}')

# 绘制新特征随时间戳的变化情况
plt.figure(figsize=(15, 6))
plt.scatter(data.index, data['dc_bus_volt_cluster_1'], s=5, color='blue', label='DC Bus Voltage Cluster 1')
plt.scatter(data.index, data['dc_bus_volt_cluster_2'], s=5, color='red', label='DC Bus Voltage Cluster 2')
plt.title('Scatter Plot of Separated DC Bus Voltage Clusters Over Time')
plt.xlabel('Time')
plt.ylabel('DC Bus Voltage')
plt.legend()
plt.ylim(0, 700) # 限制y轴范围以便观察
plt.show()
plt.savefig('ABB_电压双模态聚类分布.png')

features_to_show = ['dc_bus_volt_cluster_1', 'dc_bus_volt_cluster_2']

for feature in features_to_show:
    # 绘制平滑后的特征随时间的变化
    plt.figure(figsize=(15, 5))
#     for feature in features_to_smooth:
#         plt.plot(data['time'], data[feature + '_ema'], label=feature + ' (EMA Smoothed)')
    plt.scatter(data['timestamp'], data[feature], label=feature)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'ABB_电压双模态{feature}分布.png')


# 定义创建滞后特征和滑动窗口统计量的函数
def create_lagged_features(df, features, lag=1):
    for feature in features:
        df[feature + '_lag' + str(lag)] = df[feature].shift(lag)
    return df

def create_rolling_statistics(df, features, window=5):
    for feature in features:
        df[feature + '_rolling_mean' + str(window)] = df[feature].rolling(window=window).mean()
        df[feature + '_rolling_std' + str(window)] = df[feature].rolling(window=window).std()
    return df

# 选择除了指定特征之外的其他特征
features_to_exclude = ['igbt_junction_temp_ave' ,'timestamp']
other_features = [feature for feature in data.columns if feature not in features_to_exclude]

# 创建1步滞后特征
data = create_lagged_features(data, other_features, lag=1)

# 创建滑动窗口大小为5的移动平均和标准差
data = create_rolling_statistics(data, other_features, window=4)

