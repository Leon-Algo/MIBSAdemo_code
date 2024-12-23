############################################################################################################################################################################
###模型预测
from keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, concatenate, Input
from keras.regularizers import l2
from keras.utils import plot_model
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, Bidirectional, GRU, BatchNormalization
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 将字体设置为 Arial Unicode MS

enn_model = load_model('ABB_enn.h5')
ernn_model = load_model('ABB_ernn.h5')

# # 加载标准化器
# scaler1 = joblib.load('scaler1.pkl')
# scaler2 = joblib.load('scaler2.pkl')

new_data = pd.read_csv('processed_drive_data_na0.csv')

# # 转换timestamp列为datetime类型
# new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], errors='coerce')

# 删除Unnamed: 0列
if 'Unnamed: 0' in new_data.columns:
    new_data = new_data.drop('Unnamed: 0', axis=1)
    

# 为了更好地理解数据，我们可以选择几个特征进行可视化。
features_to_plot = ['ambient_temp', 'dc_bus_volt', 'motor_current',
                         'motor_power', 'motor_speed', 'motor_torque', 'igbt_junction_temp']


# 基于IQR方法计算异常值的范围
Q1 = new_data.iloc[:, 1:].quantile(0.25)
Q3 = new_data.iloc[:, 1:].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值的范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 剔除异常值
new_data = new_data[((new_data.iloc[:, 1:] >= lower_bound) & (new_data.iloc[:, 1:] <= upper_bound)).all(axis=1)]

# 为了减少时间序列数据中的噪声并更好地捕捉到数据的趋势，我们可以使用指数加权移动平均（EWMA）方法对数据进行平滑处理。这里，我们选择一个合理的窗口大小，如30（代表30分钟），进行平滑处理。
window_size = 30

# 使用EWMA进行平滑处理
smoothed_data = new_data[features_to_plot].ewm(span=window_size).mean()


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
X_new = supervised_data.drop(columns=[f'igbt_junction_temp(t)']).values
y_new = supervised_data[f'igbt_junction_temp(t)'].values.reshape(-1,1)

# 数据标准化
scaler1 = StandardScaler()
scaler2 = StandardScaler()
X_new_scaled = scaler1.fit_transform(X_new)
y_new_scaled = scaler2.fit_transform(np.asarray(y_new).reshape(-1, 1))

# Predict using the loaded model
y_pred_new = enn_model.predict(X_new_scaled)

## 将预测值与真实值进行反标准化处理
y_new_scaled_inverse = scaler2.inverse_transform(y_new_scaled)
y_pred_new_inverse = scaler2.inverse_transform(y_pred_new)

# 创建图形窗口
fig = plt.figure()
# plt.plot(y_new_scaled_inverse[100:160])
# plt.plot(y_pred_new_inverse[100:160])
plt.plot(y_new_scaled_inverse)
plt.plot(y_pred_new_inverse)
plt.legend(['True', 'Prediction'])
plt.title('enn_pre_set.png')
plt.xlabel('Time')
# 显示图形窗口
plt.show(fig)
# 保存图形
plt.savefig('enn_pre_set.png')

r2_test = r2_score(y_new_scaled_inverse, y_pred_new_inverse)
mse_test = mean_squared_error(y_new_scaled_inverse, y_pred_new_inverse)

print("enn新数据集上的R Squared值：", r2_test)
print("enn新数据集上的均方误差（MSE）：", mse_test)


# 对于 ERNN 模型，还需要调整输入的形状
X_new_scaled_rnn = np.reshape(X_new_scaled, (X_new_scaled.shape[0], X_new_scaled.shape[1], 1))

# 使用 ERNN 进行预测
ernn_predictions = ernn_model.predict([X_new_scaled_rnn, X_new_scaled])

# 反标准化预测结果和真实结果
ernn_predictions_inverse = scaler2.inverse_transform(ernn_predictions)
y_new_scaled_inverse = scaler2.inverse_transform(y_new_scaled)

# 现在可以使用 ernn_predictions_inverse 进行进一步的分析或可视化
# 创建图形窗口
fig = plt.figure()
plt.plot(y_new_scaled_inverse)
plt.plot(ernn_predictions_inverse)
plt.legend(['True', 'Prediction'])
plt.title('ernn_pre_set.png')
plt.xlabel('Time')
# 显示图形窗口
plt.show(fig)
# 保存图形
plt.savefig('ernn_pre_set.png')

r2_test = r2_score(y_new_scaled_inverse, ernn_predictions_inverse)
mse_test = mean_squared_error(y_new_scaled_inverse, ernn_predictions_inverse)

print("ernn新数据集上的R Squared值：", r2_test)
print("ernn新数据集上的均方误差（MSE）：", mse_test)


### xgboost模型预测
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载模型
loaded_model = xgb.XGBRegressor()
loaded_model.load_model('xgb_model.pkl')

# 2. 使用模型进行预测
xgb_predictions = loaded_model.predict(X_new).reshape(-1, 1)


# 反标准化预测结果和真实结果
xgb_predictions_inverse = scaler2.inverse_transform(xgb_predictions)
y_new_scaled_inverse = scaler2.inverse_transform(y_new_scaled)

# 3. 评估预测效果（可选）
xgb_r2_value = r2_score(y_new_scaled_inverse, xgb_predictions_inverse)
xgb_mse_value = mean_squared_error(y_new_scaled_inverse, xgb_predictions_inverse)



### 最后，使用三个模型的预测结果进行加权平均来得到最终的预测.
# 模型预测
enn_predictions = enn_model.predict(X_new_scaled)
xgb_predictions = xgb_model.predict(X_new_scaled).reshape(-1,1)
ernn_predictions = ernn_model.predict([X_new_scaled_rnn, X_new_scaled])

if xgb_r2_value<0:
    # 加权平均
    final_predictions = 0.6 * ernn_predictions + 0.4 * enn_predictions 
    final_predictions = scaler2.inverse_transform(final_predictions)

    # 如果需要，可以将预测结果保存到CSV文件
    pd.DataFrame(final_predictions, columns=['final_predictions']).to_csv('final_predictions.csv', index=False)


    r2_test = r2_score(y_new_scaled_inverse, final_predictions)
    mse_test = mean_squared_error(y_new_scaled_inverse, final_predictions)

    print("2新数据集上的R Squared值：", r2_test)
    print("新数据集上的均方误差（MSE）：", mse_test)    
else:
    # 加权平均
    final_predictions = 0.4 * ernn_predictions + 0.25 * enn_predictions + 0.35 * xgb_predictions
    final_predictions = scaler2.inverse_transform(final_predictions)

    # 如果需要，可以将预测结果保存到CSV文件
    pd.DataFrame(final_predictions, columns=['final_predictions']).to_csv('final_predictions.csv', index=False)


    r2_test = r2_score(y_new_scaled_inverse, final_predictions)
    mse_test = mean_squared_error(y_new_scaled_inverse, final_predictions)

    print("3新数据集上的R Squared值：", r2_test)
    print("新数据集上的均方误差（MSE）：", mse_test)
    
# 绘制实际值与预测值的对比图
plt.figure(figsize=(15, 6))
plt.plot(y_new, label='Actual')
plt.plot(final_predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.ylabel('igbt_junction_temp')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# 保存图形
plt.savefig('新数据集最终预测拟合结果.png')