
    
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# 文件路径和编码
file_path = 'E:/13届市调分析大赛(研究生)/调查问卷数据处理后2.csv'
file_encoding = 'ansi'

# 读取数据
data = pd.read_csv(file_path, encoding=file_encoding)



# 将数据集分为特征变量和目标变量
X = data.drop('ai_search_user_expectation', axis=1)
y = data['ai_search_user_expectation']

# 创建逻辑回归模型
model = LogisticRegression()

# 使用逐步回归方法进行变量选择
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)

# 输出选中的特征变量
selected_features = []
for i in range(len(X.columns)):
    if rfe.support_[i]:
        selected_features.append(X.columns[i])
print("Selected Features: ", selected_features)

# 计算所选变量的AIC值
X_train = sm.add_constant(X[selected_features])
model = sm.Logit(y, X_train).fit()
aic_value = model.aic
print("Selected Variables' AIC: ", aic_value)

# 合并特征变量名和AIC值为一个表格并输出
result_table = pd.DataFrame({"Selected Features": selected_features, "AIC Value": aic_value})
print(result_table)




# # 将数据集分为特征变量和目标变量
# X = data.drop('ai_search_user_expectation', axis=1)
# y = data['ai_search_user_expectation']

# # 创建逻辑回归模型
# model = LogisticRegression(multi_class='multinomial')

# # 使用逐步回归方法进行变量选择
# rfe = RFE(model, n_features_to_select=5)
# rfe.fit(X, y)

# # 输出选中的特征变量
# selected_features = []
# for i in range(len(X.columns)):
#     if rfe.support_[i]:
#         selected_features.append(X.columns[i])
# print("Selected Features: ", selected_features)

# # 计算所选变量的AIC值
# X_train = sm.add_constant(X[selected_features])
# model = sm.MNLogit(y, X_train).fit()
# aic_value = model.aic
# print("Selected Variables' AIC: ", aic_value)

# # 合并特征变量名和AIC值为一个表格并输出
# result_table = pd.DataFrame({"Selected Features": selected_features, "AIC Value": aic_value})
# print(result_table)
