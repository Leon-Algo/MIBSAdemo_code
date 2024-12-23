import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
from eli5.sklearn import PermutationImportance

# 文件路径和编码
file_path = 'E:/13届市调分析大赛(研究生)/调查问卷数据处理后2.csv'
file_encoding = 'ansi'

# 读取数据
def load_data():
    df = pd.read_csv(file_path, encoding=file_encoding)
    X = df.drop('ai_search_user_expectation', axis=1)
    y = df['ai_search_user_expectation']
    return X, y

# 训练集和测试集的拆分
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 创建和训练 XGBoost 模型
def train_model(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    xgb_model = xgb.XGBClassifier()
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# 预测并评估模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print('准确率：{:.2f}，精确度：{:.2f}，召回率：{:.2f}，F1 分数：{:.2f}'.format(accuracy, precision, recall, f1))

# 可视化特征重要性分数
def plot_feature_importances(model):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax)
    plt.show()

# 特征重要性分析
def feature_importance_analysis(model, X_test, y_test):
    perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
    feature_importance = pd.DataFrame(perm.feature_importances_, index=X_test.columns, columns=['重要性']).sort_values(by='重要性', ascending=False)
    return feature_importance

# 主函数
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_feature_importances(model)
    feature_importance = feature_importance_analysis(model, X_test, y_test)
    print(feature_importance)

if __name__ == '__main__':
    main()







# import pandas as pd
# import xgboost as xgb
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from eli5.sklearn import PermutationImportance
# import eli5
# from IPython.display import display


# # 文件路径和编码
# file_path = 'E:/13届市调分析大赛(研究生)/调查问卷数据处理后2.csv'
# file_encoding = 'ansi'

# # 读取数据
# def load_data():
#     df = pd.read_csv(file_path, encoding=file_encoding)
#     X = df.drop('ai_search_user_expectation', axis=1)
#     y = df['ai_search_user_expectation']
#     return X, y

# # 训练集和测试集的拆分
# def split_data(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     return X_train, X_test, y_train, y_test

# # 创建和训练 XGBoost 模型
# def train_model(X_train, y_train):
#     param_grid = {
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.1, 0.3]
#     }
#     xgb_model = xgb.XGBClassifier()
#     grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5)
#     grid_search.fit(X_train, y_train)
#     return grid_search.best_estimator_

# # 预测并评估模型
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='macro')
#     recall = recall_score(y_test, y_pred, average='macro')
#     f1 = f1_score(y_test, y_pred, average='macro')
#     print('准确率：{:.2f}，精确度：{:.2f}，召回率：{:.2f}，F1 分数：{:.2f}'.format(accuracy, precision, recall, f1))

# # 可视化特征重要性分数
# def plot_feature_importances(model):
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#     fig, ax = plt.subplots(figsize=(10, 8))
#     xgb.plot_importance(model, ax=ax)
#     plt.show()

# # 特征重要性分析
# def feature_importance_analysis(model, X_test, y_test):
#     perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
#     feature_importance = pd.DataFrame({
#         '特征重要性': perm.feature_importances_,
#         '可靠性得分': perm.feature_importances_std_
#     }, index=X_test.columns).sort_values(by='特征重要性', ascending=False)
#     display(eli5.show_weights(perm, feature_names=X_test.columns.tolist()))
#     return feature_importance

# # 主函数
# def main():
#     X, y = load_data()
#     X_train, X_test, y_train, y_test = split_data(X, y)
#     model = train_model(X_train, y_train)
#     evaluate_model(model, X_test, y_test)
#     plot_feature_importances(model)
#     feature_importance = feature_importance_analysis(model, X_test, y_test)
#     print(feature_importance)

# if __name__ == '__main__':
#     main()















# # 划分训练集和测试集
# X = df.drop('ai_search_expectation', axis=1)
# y = df['ai_search_expectation']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # 训练xgboost模型
# model = xgb.XGBClassifier(objective='multi:softmax', num_class=5, seed=42)
# model.fit(X_train, y_train)

# # 在测试集上进行预测
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')


# # 计算特征重要性得分
# plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文
# plt.rcParams['axes.unicode_minus']=False   # 用来正常显示负号

# plot_importance(model)
# plt.show()













# #问卷数据导入
# data = pd.read_excel('E:/13届市调分析大赛(研究生)/processed_data.csv')


# #数据预处理
# ##对于单选题，使用LabelEncoder将选项转换为数字
# ##对于多选题，将选项转换为二进制数，每个选项对应一个二进制位，选中则为1，不选则为0
# # 将单选题选项转换为数字
# le = LabelEncoder()
# data['性别'] = le.fit_transform(data['性别'])
# data['年龄'] = le.fit_transform(data['年龄'])
# data['教育经历'] = le.fit_transform(data['教育经历'])
# data['职业'] = le.fit_transform(data['职业'])
# data['使用频率'] = le.fit_transform(data['使用频率'])
# data['传统搜索满意度'] = le.fit_transform(data['传统搜索满意度'])
# data['信任度'] = le.fit_transform(data['信任度'])
# data['了解度'] = le.fit_transform(data['了解度'])
# data['简便性重视程度'] = le.fit_transform(data['简便性重视程度'])
# data['规范性重视程度'] = le.fit_transform(data['规范性重视程度'])
# data['页面美观性重视程度'] = le.fit_transform(data['页面美观性重视程度'])
# data['愿意与搜索引擎沟通'] = le.fit_transform(data['愿意与搜索引擎沟通'])
# data['愿意使用信息提取/总结的搜索引擎'] = le.fit_transform(data['愿意使用信息提取/总结的搜索引擎'])
# data['AI搜索会使得搜索到的内容和价值更丰富'] = le.fit_transform(data['AI搜索会使得搜索到的内容和价值更丰富'])
# data['AI搜索会更符合用户期待'] = le.fit_transform(data['AI搜索会更符合用户期待'])

# # 将多选题选项转换为二进制数
# data['主要使用搜索引擎解决哪类问题1'] = data['主要使用搜索引擎解决哪类问题'].str.contains('找资料').astype(int)
# data['主要使用搜索引擎解决哪类问题2'] = data['主要使用搜索引擎解决哪类问题'].str.contains('看新闻').astype(int)
# data['主要使用搜索引擎解决哪类问题3'] = data['主要使用搜索引擎解决哪类问题'].str.contains('听歌、看视频或娱乐相关').astype(int)
# data['主要使用搜索引擎解决哪类问题4'] = data['主要使用搜索引擎解决哪类问题'].str.contains('查询作业答案').astype(int)
# data.drop('主要使用搜索引擎解决哪类问题', axis=1, inplace=True)

# #保存预处理数据
# data.to_csv('E:/13届市调分析大赛(研究生)/预处理后的数据.csv', index=False)


# #将数据分为特征和标签
# X = data.drop('AI搜索会更符合用户期待', axis=1)
# y = data['AI搜索会更符合用户期待']

