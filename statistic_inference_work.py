from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# import jieba
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error,classification_report,roc_auc_score
from sklearn.ensemble import RandomForestClassifier


from datetime import datetime
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn import preprocessing

# 导入数据
data = pd.read_csv("E:\\QQ\\研究生课程小作业\\统计推断小作业\\bank-full.csv",sep=';')



### 数据预处理(定性数据进行编码处理；定量数据进行离散化和标准化处理) 

# 二分类变量进行0-1编码
def encode_bin_attrs(data, bin_attrs):    
    for i in bin_attrs:
        data.loc[data[i] == 'no', i] = 0
        data.loc[data[i] == 'yes', i] = 1
    return data

# 有序变量进行1-2-3有序编码
def encode_edu_attrs(data):
    values = ["primary", "secondary", "tertiary"]
    levels = range(1,len(values)+1)
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['education'] == v, 'education'] = dict_levels[v]
    return data

# 无序变量转哑变量进行0-1编码
def encode_cate_attrs(data, cate_attrs):
    data = encode_edu_attrs(data)
    cate_attrs.remove('education')
    for i in cate_attrs:
        dummies_df = pd.get_dummies(data[i])
        dummies_df = dummies_df.rename(columns=lambda x: i+'_'+str(x))
        data = pd.concat([data,dummies_df],axis=1)
        data = data.drop(i, axis=1)
    return data

# 数值型特征标准化处理
def feature_scaling(data, numeric_attrs):
    for i in numeric_attrs:
        std = data[i].std()
        if std != 0:
            data[i] = (data[i]-data[i].mean()) / std
        else:
            data = data.drop(i, axis=1)
    return data

# 数值型变量(特征)离散化和标准化
def trans_num_attrs(data, numeric_attrs):
    bining_num = 10
    bining_attr = 'age'
    # 对连续变量(特征)进行划区间分组离散化---特征分区离散话可以提高模型对该特征的鲁棒性
    # (决策树、朴素贝叶斯等算法，都是基于离散型的数据展开的，不处理的话就是按连续样本个数分类这会使得分类决策模型运算时间巨长)
    # (如果要使用该类算法，必须将离散型的数据进行。有效的离散化能减小算法的时间和空间开销，提高系统对样本的分类聚类能力和抗噪声能力。)
    data[bining_attr] = pd.qcut(data[bining_attr], bining_num)
    data[bining_attr] = pd.factorize(data[bining_attr])[0]+1
    # # 对数值型数据进行标准化处理
    # for i in numeric_attrs: 
    #     scaler = preprocessing.StandardScaler()
    #     data[i] = scaler.fit_transform(data[i])
    return data

# 随机森林分类器预测(unknown)
def train_predict_unknown(trainX, trainY, testX):
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainX, trainY)
    test_predictY = forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY,index=testX.index)

# 使用数据完整的样本对缺失部分特征值的样本进行预测填充
def fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs):
    # fill_attrs = ['education', 'default', 'housing', 'loan']
    fill_attrs = []
    for i in bin_attrs+cate_attrs:
        if data[data[i] == 'unknown']['y'].count() < 500:
            # delete col containing unknown
            data = data[data[i] != 'unknown'] 
        else:
            fill_attrs.append(i)
    
    data = encode_cate_attrs(data, cate_attrs)
    data = encode_bin_attrs(data, bin_attrs)
    data = trans_num_attrs(data, numeric_attrs)
    data['y'] = data['y'].map({'no': 0, 'yes': 1}).astype(int)
    for i in fill_attrs:  
        test_data = data[data[i] == 'unknown']
        testX = test_data.drop(fill_attrs, axis=1)
        train_data = data[data[i] != 'unknown']        
        trainY = train_data[i]
        trainX = train_data.drop(fill_attrs, axis=1)    
        test_data[i] = train_predict_unknown(trainX, trainY, testX)
        data = pd.concat([train_data, test_data])
    
    return data

def preprocess_data():
    input_data_path = "E:\\QQ\\研究生课程小作业\\统计推断小作业\\bank-full.csv"
    processed_data_path = 'E:\\QQ\\研究生课程小作业\\统计推断小作业\\bank-full333.csv'
    print("Loading data...")
    data = pd.read_csv(input_data_path, sep=';')
    print("Preprocessing data...")
    numeric_attrs = ['age', 'duration', 'campaign', 'pdays', 'previous']
    bin_attrs = ['default', 'housing', 'loan']
    cate_attrs = ['poutcome', 'education', 'job', 'marital','contact', 'month']
    
    data = shuffle(data)
    data = fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs)
    data.to_csv(processed_data_path, index=False)


start_time = datetime.now()
preprocess_data()
end_time = datetime.now()
delta_seconds = (end_time - start_time).seconds
print("Cost time: {}s".format(delta_seconds))
