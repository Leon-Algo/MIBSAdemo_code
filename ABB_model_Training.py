############################################################################################################################################################################
###模型训练
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, concatenate, Input
from keras.regularizers import l2
from keras.utils import plot_model
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss, Reduction
import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, Bidirectional, GRU, BatchNormalization
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 将字体设置为 Arial Unicode MS

# 加载数据
# data = pd.read_excel('/kaggle/input/abb-competition/data1_final(absolution)_cleaned_data.xlsx')
data = pd.read_csv('processed_drive_data_na0.csv')
# data = pd.read_csv('/kaggle/input/abb-competition/processed_drive_data_na0_half.csv')
# data = pd.read_csv('/kaggle/input/abb-competition/processed_drive_data_na0_smallpart.csv')

# 转换timestamp列为datetime类型
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

# 删除Unnamed: 0列
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)
    

# 为了更好地理解数据，我们可以选择几个特征进行可视化。
features_to_plot = ['ambient_temp', 'dc_bus_volt', 'motor_current','motor_power', 'motor_speed', 'motor_torque', 'igbt_junction_temp']

# 基于IQR方法计算异常值的范围
Q1 = data.iloc[:, 1:].quantile(0.25)
Q3 = data.iloc[:, 1:].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值的范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 剔除异常值
data = data[((data.iloc[:, 1:] >= lower_bound) & (data.iloc[:, 1:] <= upper_bound)).all(axis=1)]

# 显示更新后的数据的维度
data.shape

# 为了减少时间序列数据中的噪声并更好地捕捉到数据的趋势，我们可以使用指数加权移动平均（EWMA）方法对数据进行平滑处理。这里，我们选择一个合理的窗口大小，如30（代表30分钟），进行平滑处理。
window_size = 30

# 使用EWMA进行平滑处理
smoothed_data = data[features_to_plot].ewm(span=window_size).mean()

# 这里，我们选择一个窗口大小，如10，将过去10个时间点的数据作为输入，当前时间点的`igbt_junction_temp(5.11)`作为输出。
def series_to_supervised(data, features, n_in=1, n_out=1, dropnan=True):
    n_vars = len(features)
    df = data[features]
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'{feature}(t-{i})' for feature in features]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'{feature}(t)' for feature in features]
        else:
            names += [f'{feature}(t+{i})' for feature in features]
    # 将所有数据合并到一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 去除NaN行
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 定义窗口大小
n_hours = 10

# 使用窗口大小为10的数据作为输入，当前时间点的数据作为输出
features_for_model = smoothed_data.columns.tolist()
supervised_data = series_to_supervised(smoothed_data, features_for_model, n_hours, 1)

# 丢弃我们不想预测的列
# supervised_data.drop(supervised_data.columns[-7:-1], axis=1, inplace=True)

# 显示转化后的数据
supervised_data.head()

from sklearn.model_selection import train_test_split

# 定义输入和输出
X = supervised_data.drop(columns=[f'igbt_junction_temp(t)']).values
y = supervised_data[f'igbt_junction_temp(t)'].values.reshape(-1,1)

# 数据标准化
scaler1 = StandardScaler()
scaler2 = StandardScaler()
X = scaler1.fit_transform(X)
y = scaler2.fit_transform(np.asarray(y).reshape(-1, 1))

# 划分训练集和测试集
X, X_test, y, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=False)

print(X.shape, X_test.shape, y.shape, y_test.shape)


### 构建新的基于物理信息约束的loss函数
class CustomLoss(Loss):
    def __init__(self, xt_train, input_tensor, **kwargs):
        super().__init__(**kwargs)
        self.xt_train = xt_train
        self.input_tensor = input_tensor
        self.alpha = 0.1
        self.gamma = 10

    def call(self, y_true, y_pred):
        # 使用索引获取对应的xt_batch
        indices = self.input_tensor[1]  # 获取输入中的索引
        xt_batch = tf.gather(self.xt_train, indices)

        y_diff = y_true - y_pred
        mse = K.mean(K.square(y_diff))
        
        re_loss = K.square(K.relu(xt_batch - y_pred)) 
        re_loss_mean = self.gamma * K.mean(re_loss)
        
        return (1 - self.alpha) * mse + self.alpha * re_loss_mean


### enn训练及评价
# Function to create a DNN model
#dnn1
input_1 = Input(shape=(X.shape[1]))
# hlay1_1 = Dense(16,activation = 'relu')(input_1)
hlay1_1 = Dense(32,activation = 'relu')(input_1)
drop1 = Dropout(0.2)(hlay1_1)
hlay2_1 = Dense(32,activation = 'relu')(drop1)
drop2 = Dropout(0.2)(hlay2_1)
hlay3_1 = Dense(32,activation = 'relu')(drop2)
drop3 = Dropout(0.2)(hlay3_1)
hlay4_1 = Dense(16,activation = 'relu')(drop3)
drop4 = Dropout(0.2)(hlay4_1)
final_dnn1 = Dense(1,activation = 'linear')(drop4)
# hlay2_1 = Dense(16,activation = 'relu')(hlay1_1)
# final_dnn1 = Dense(1,activation = 'linear')(hlay2_1)
dnn1 = Model(inputs=input_1, outputs=final_dnn1)
dnn1.compile(optimizer = "adam", loss = "mean_squared_error")

#dnn2
input_2 = Input(shape=(X.shape[1]))
hlay1_2 = Dense(64,activation = 'relu')(input_2)
drop1 = Dropout(0.2)(hlay1_2)
hlay2_2 = Dense(32,activation = 'relu')(drop1)
drop2 = Dropout(0.2)(hlay2_2)
hlay3_2 = Dense(16,activation = 'relu')(hlay2_2)
drop3 = Dropout(0.2)(hlay3_2)
final_dnn2 = Dense(1,activation = 'linear')(drop3)
dnn2 = Model(inputs=input_2, outputs=final_dnn2)
dnn2.compile(optimizer = "adam", loss = "mean_absolute_percentage_error")

#dnn3
input_3 = Input(shape=(X.shape[1]))
hlay1_3 = Dense(16,activation = 'sigmoid')(input_3)
drop1 = Dropout(0.2)(hlay1_2)
hlay2_3 = Dense(16,activation = 'sigmoid')(hlay1_3)
hlay3_3 = Dense(8,activation = 'relu')(hlay2_3)
final_dnn3 = Dense(1,activation = 'linear')(hlay3_3)
dnn3 = Model(inputs=input_3, outputs=final_dnn3)
dnn3.compile(optimizer = "adam", loss = "mean_squared_error")

#dnn4
input_4 = Input(shape=(X.shape[1]))
hlay1_4 = Dense(32,activation = 'relu')(input_4)
hlay2_4 = Dense(32,activation = 'sigmoid')(hlay1_4)
final_dnn4 = Dense(1,activation = 'linear')(hlay2_4)
dnn4 = Model(inputs=input_4, outputs=final_dnn4)
dnn4.compile(optimizer = "adam", loss = "mean_squared_error")

#weightedNN
# 定义权重
weights = {
    'motor_current(1.07)': 0.25,
    'motor_power_ave(1.15)': 0.20,
    'motor_torque_ave(1.1)': 0.18,
    'ambient_temp（1.31)': 0.15,
    'motor_speed_ave(1.01)': 0.12,
    'dc_bus_volt(1.11)': 0.10
}

# 创建列名与列索引的映射
column_mapping = {    'motor_current(1.07)': 0,
    'motor_power_ave(1.15)': 1,
    'motor_torque_ave(1.1)': 2,
    'ambient_temp（1.31)': 3,
    'motor_speed_ave(1.01)': 4,
    'dc_bus_volt(1.11)': 5}  # 列名与列索引的映射
column_index = column_mapping['ambient_temp（1.31)']  # 获取列索引


# 输入层
# input_weighted = Input(shape=(X.shape[1]))
input_weighted_data_5 = Input(shape=(X_train.shape[1],))
input_weighted_indices_5 = Input(shape=(1,))

# 分别为每个特征定义一个分支
branch_outputs = []

for i, feature in enumerate(weights.keys()):
    # 为每个特征定义一个Dense层，神经元数量与权重成正比
    neurons = int(32 * weights[feature])
#     branch = Dense(neurons, activation='relu')(input_weighted[:, i:i+1])
    branch = Dense(neurons, activation='relu')(input_weighted_data_5[:, column_mapping[feature]:column_mapping[feature]+1])
    branch = Dropout(0.2)(branch)
    branch = Dense(neurons//2, activation='relu')(branch)
    branch_outputs.append(branch)

# 将所有的分支合并
merged = concatenate(branch_outputs)

# 为合并后的输出定义额外的Dense层
dense = Dense(64, activation='relu')(merged)
dense = Dropout(0.5)(dense)
final_weighted = Dense(1)(dense)

# model_weighted = Model(inputs=input_weighted, outputs=final_weighted)
model_weighted = Model(inputs=[input_weighted_data_5, input_weighted_indices_5], outputs=final_weighted)
model_weighted.compile(optimizer = "adam", loss = "mean_squared_error")


### 分别训练独立的全连接网络，为了进一步提高集合中神经网络组成部分的多样性，我们从整体训练数据中对每个网络进行随机采样。这是一种套袋方法。
# X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.6, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.3, shuffle=False)
dnn1.fit(X_train, y_train, epochs = 15, batch_size = 32,verbose=True,validation_data=(X_val, y_val))

# X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.6, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.3, shuffle=False)
dnn2.fit(X_train, y_train, epochs = 15, batch_size = 32,verbose=True,validation_data=(X_val, y_val))

# X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.6, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.3, shuffle=False)
dnn3.fit(X_train, y_train, epochs = 15, batch_size = 32,verbose=True,validation_data=(X_val, y_val))

# X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.6, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.3, shuffle=False)
dnn4.fit(X_train, y_train, epochs = 15, batch_size = 64,verbose=True,validation_data=(X_val, y_val))

# Freeze the DNN models
dnn1.trainable = False
dnn2.trainable = False
dnn3.trainable = False
dnn4.trainable = False

# Ensemble Neural Network (ENN)
enn_input = Input(shape=(X_train.shape[1]))
const_1 = dnn1(enn_input)
const_2 = dnn2(enn_input)
const_3 = dnn3(enn_input)
const_4 = dnn4(enn_input)
enn_1 = Dense(2, activation='relu')(enn_input)
merge = concatenate(inputs=[const_1, const_2, const_3, const_4, enn_1])
enn_2 = Dense(32, activation='relu')(merge)
enn_3 = Dense(32, activation='relu')(enn_2)
final = Dense(1, activation='linear')(enn_3)
enn = Model(inputs=enn_input, outputs=final)
enn.compile(optimizer="adam", loss="mean_squared_error")

# Train the ENN
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
# 给当前模型设置早停和学习率调度回调
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)
history = enn.fit(X_train, y_train, epochs=100, batch_size=32, verbose=True, validation_data=(X_val, y_val),callbacks=[early_stop, reduce_lr])
# 保存模型
enn.save('ABB_enn.h5')
# 保存模型结构图
plot_model(enn, to_file='ABB_enn.png', show_shapes=True)

# Plot the training history (Plot 2)
# 创建第二个图形窗口
fig2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(['Training', 'Validation'])
# 显示第二个图形窗口
plt.show(fig2)
# 保存第二个图形窗口
fig2.savefig('ABB_plot2_enn_training_history.png')


# 在enn测试集上评价模型
y_pred_test = enn.predict(X_test)
## 将预测值与真实值进行反标准化处理
y_test_inverse = scaler2.inverse_transform(y_test)
y_pred_test_inverse = scaler2.inverse_transform(y_pred_test)
# 创建第四个图形窗口
fig4 = plt.figure(4)
plt.plot(y_test_inverse)
plt.plot(y_pred_test_inverse)
plt.legend(['True', 'Prediction'])
plt.title('ABB_plot4_enn_test_set.png')
plt.xlabel('timestamp')
# 显示第四个图形窗口
plt.show(fig4)
# 保存第四个图形窗口
fig4.savefig('ABB_plot4_enn_test_set.png')  # Plot 4

r2_test = r2_score(y_test_inverse, y_pred_test_inverse)
mse_test = mean_squared_error(y_test_inverse, y_pred_test_inverse)
print("enn测试集上的R Squared值：", r2_test)
print("enn测试集上的均方误差（MSE）：", mse_test)



### ernn训练及评价
### 为了更好地捕捉趋势，我们将把 RNN 纳入我们的模型中。这将使我们能够按原样处理数据——时间序列数据。希望添加的上下文能让我们防止这些错误的峰值
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, shuffle=False)

X_train_rnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
y_train_rnn = np.reshape(y_train, (y_train.shape[0],1))
X_val_rnn = np.reshape(X_val, (X_val.shape[0], X_val.shape[1],1))
y_val_rnn = np.reshape(y_val, (y_val.shape[0],1))
# X_train_rnn = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))
# y_train_rnn = np.reshape(y_train, (y_train.shape[0],1))
# X_val_rnn = np.reshape(X_val, (X_val.shape[0],1, X_val.shape[1]))
# y_val_rnn = np.reshape(y_val, (y_val.shape[0],1))

enn.trainable = False;
# 使用 LSTM 替代 RNN
input_enn = Input(shape=(X_train.shape[1]))
input_rnn = Input(shape=(X_train.shape[1], 1))
# input_rnn = Input(shape=(1, X_train.shape[1]))
lstm1 = LSTM(32, return_sequences=True)(input_rnn)
drop1 = Dropout(0.2)(lstm1)
lstm2 = LSTM(64, return_sequences=True)(drop1)
batchnorm1 = BatchNormalization()(lstm2)
drop2 = Dropout(0.3)(batchnorm1)
lstm3 = LSTM(128, return_sequences=False)(drop2)
drop3 = Dropout(0.4)(lstm3)
dense1 = Dense(64, activation='relu')(drop3)
drop4 = Dropout(0.5)(dense1)

enn_lay = enn(input_enn)
merge = concatenate(inputs = [enn_lay,drop4])
combine1 = Dense(16,activation = 'relu')(merge)
combine2 = Dense(16,activation = 'relu')(combine1)
final = Dense(1)(combine2)

ernn = Model(inputs=[input_rnn,input_enn], outputs=final)
ernn.compile(optimizer = "adam", loss = "mean_squared_error")
plot_model(ernn)
# history = ernn.fit([X_train_rnn,X_train], y_train, epochs = 50, batch_size = 32,verbose=True,validation_data=([X_val_rnn,X_val], y_val))
# 给当前模型设置早停和学习率调度回调
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)
history = ernn.fit([X_train_rnn,X_train], y_train, epochs = 50, batch_size = 32,verbose=True,validation_data=([X_val_rnn,X_val], y_val),callbacks=[early_stop, reduce_lr])
# 保存模型
ernn.save('ABB_ernn.h5')
# 保存模型结构图
plot_model(ernn, to_file='ABB_ernn.png', show_shapes=True, show_layer_names=True)






### 可视化训练过程效果--在训练集和验证集上看MSE
# 创建第五个图形窗口
fig5 = plt.figure(5)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ABB_plot5_ernn_training_history')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(['Training','Validation'])
# 显示第五个图形窗口
plt.show(fig5)
# 保存第五个图形窗口
fig5.savefig('ABB_plot5_ernn_training_history.png')  # Plot 5



### 在ernn训练集上评价模型
y_pred_train = ernn.predict([X_train_rnn,X_train])
## 将预测值与真实值进行反标准化处理
y_train_inverse = scaler2.inverse_transform(y_train)
y_pred_train_inverse = scaler2.inverse_transform(y_pred_train)

r2_train = r2_score(y_train_inverse, y_pred_train_inverse)
mse_train = mean_squared_error(y_train_inverse, y_pred_train_inverse)
print("ernn训练集上的R Squared值：", r2_train)
print("ernn训练集上的均方误差（MSE）：", mse_train)



### 在ernn测试集上评价模型
X_test_rnn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
# X_test_rnn = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))

y_pred_test = ernn.predict([X_test_rnn,X_test])  ####rnn的test数据集和普通test数据集一起输入
## 将预测值与真实值进行反标准化处理
y_test_inverse = scaler2.inverse_transform(y_test)
y_pred_test_inverse = scaler2.inverse_transform(y_pred_test)

# 创建第七个图形窗口
fig7 = plt.figure(7)
plt.plot(y_test_inverse)
plt.plot(y_pred_test_inverse)
plt.legend(['True','Prediction'])
plt.title('ABB_plot7_ernn_test_set')
plt.xlabel('timestamp')

# 显示第七个图形窗口
plt.show(fig7)
# 保存第七个图形窗口
fig7.savefig('ABB_plot7_ernn_test_set.png')  # Plot 7

r2_test = r2_score(y_test_inverse, y_pred_test_inverse)
mse_test = mean_squared_error(y_test_inverse, y_pred_test_inverse)
print("ernn测试集上的R Squared值：", r2_test)
print("ernn测试集上的均方误差（MSE）：", mse_test)




### xgboost训练及评价
# XGBoost是一个优化的分布式梯度提升库，它具有高效、灵活和便携的特点。它在多个机器学习竞赛中都表现出色。
import xgboost as xgb

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.3, shuffle=False)
# 定义和训练XGBoost模型
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# 预测测试集
xgb_predictions = xgb_model.predict(X_test).reshape(-1,1)

# 使用r2和MSE对模型进行评价
r2_test2 = r2_score(y_test, xgb_predictions)
mse_test2 = mean_squared_error(y_test, xgb_predictions)

# 绘制实际值与预测值的对比图
plt.figure(figsize=(15, 6))
plt.plot(y_test, label='Actual')
plt.plot(xgb_predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.ylabel('igbt_junction_temp')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 保存XGBoost模型
model_filename = "xgb_model.pkl"
xgb_model.save_model(model_filename)


### 最后，使用三个模型的预测结果进行加权平均来得到最终的预测。加权可以根据每个模型在验证集上的性能来确定
# 模型预测
enn_predictions = enn.predict(X_test)
xgb_predictions = xgb_model.predict(X_test).reshape(-1,1)
ernn_predictions = ernn.predict([X_test_rnn,X_test])

# 加权平均
final_predictions = 0.4 * ernn_predictions + 0.25 * enn_predictions + 0.35 * xgb_predictions
final_predictions = y_train_inverse = scaler2.inverse_transform(final_predictions)

r2_test = r2_score(y_test_inverse, final_predictions)
mse_test = mean_squared_error(y_test_inverse, final_predictions)

print("测试集上的R Squared值：", r2_test)
print("测试集上的均方误差（MSE）：", mse_test)

# 绘制实际值与预测值的对比图
plt.figure(figsize=(15, 6))
plt.plot(y_test_inverse, label='Actual')
plt.plot(final_predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.ylabel('igbt_junction_temp')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# 保存图形
plt.savefig('ABB_测试集最终预测拟合结果.png')