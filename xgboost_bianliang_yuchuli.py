# import pandas as pd

# #问卷数据导入
# df = pd.read_excel('E:/13届市调分析大赛(研究生)/问卷结果手选版-原文字分类.xlsx')

# # 将原始列名修改为简化列名
# col_names = {
#     '您的性别:': '性别',
#     '您的年龄:': '年龄',
#     '您的教育经历是:': '教育',
#     '您的职业是:': '职业',
#     '您使用手机或电脑进行搜索的频率为:': '搜索频率',
#     '您主要使用搜索引擎解决哪类问题:': '搜索意图',
#     '传统搜索引擎可以很快解决您的检索任务': '传统搜索引擎的效率',
#     '您对现在使用的搜索引擎感到满意': '传统搜索引擎的满意度',
#     '您信任传统搜索引擎上的答案吗?': '传统搜索引擎的信任度',
#     '您了解过对话式搜索引擎嘛?(ChatGPT/文心一言/Newbing等等)': 'AI搜索引擎的了解',
#     '您对于搜索引擎的操作简便性的重视程度:': '操作简便性重视程度',
#     '您对于搜索引擎的回答的规范性的重视程度:': '回答规范性重视程度',
#     '您对于搜索引擎的页面美观性的重视程度?': '页面美观性重视程度',
#     '您愿意搜索引擎能像人一样和您沟通吗:': '对AI搜索的接受度',
#     '您对AI搜索有哪些期待?': 'AI搜索的期待项',
#     '您愿意使用可以信息提取/总结的搜索引擎吗:': 'AI搜索的使用意愿度',
#     '您是否认同AI搜索会使得搜索到的内容和价值更丰富': 'AI搜索的价值度',
#     '您是否认同AI搜索会更符合用户期待:': 'AI搜索的期待度'
# }

# df = df.rename(columns=col_names)

# # 将多选题转换为哑变量
# 搜索意图 = df['搜索意图'].str.get_dummies(sep='□')
# AI搜索的期待项 = df['AI搜索的期待项'].str.get_dummies(sep='□')

# # 将单选题转换为哑变量
# categorical_cols = [
#     '性别', '年龄', '教育', '职业', '搜索频率',
#     '传统搜索引擎的效率', '传统搜索引擎的满意度', '传统搜索引擎的信任度',
#     'AI搜索引擎的了解', '操作简便性重视程度', '回答规范性重视程度',
#     '页面美观性重视程度', '对AI搜索的接受度', 'AI搜索的使用意愿度',
#     'AI搜索的价值度', 'AI搜索的期待度'
# ]

# for col in categorical_cols:
#     df[col] = pd.Categorical(df[col])
#     df[col] = df[col].cat.codes

# # 合并哑变量和原始数据
# df = pd.concat([df, 搜索意图, AI搜索的期待项], axis=1)

# # 删除原始数据中的多选题列
# df = df.drop(['搜索意图', 'AI搜索的期待项'], axis=1)

# # 保存预处理后的数据
# df.to_csv('E:/13届市调分析大赛(研究生)/processed_data1.csv', index=False)

import pandas as pd
import numpy as np

# 读取问卷数据
df = pd.read_excel('E:/13届市调分析大赛(研究生)/问卷结果手选版-原文字分类.xlsx')

# 修改列名为简化列名
df = df.rename(columns={
    '您的性别': 'gender',
    '您的年龄': 'age',
    '您的教育经历是': 'education',
    '您的职业是': 'occupation',
    '您使用手机或电脑进行搜索的频率为': 'search_frequency',
    '您主要使用搜索引擎解决哪类问题': 'search_purpose',
    '传统搜索引擎可以很快解决您的检索任务': 'traditional_search_efficiency',
    '您对现在使用的搜索引擎感到满意': 'satisfaction',
    '您信任传统搜索引擎上的答案吗?': 'trust',
    '您了解过对话式搜索引擎嘛?(ChatGPT/文心一言/Newbing等等)': 'chatbot_search_knowledge',
    '您对于搜索引擎的操作简便性的重视程度': 'user_friendly',
    '您对于搜索引擎的回答的规范性的重视程度': 'accuracy',
    '您对于搜索引擎的页面美观性的重视程度?': 'aesthetics',
    '您愿意搜索引擎能像人一样和您沟通吗': 'chatbot_search_preference',
    '您对AI搜索有哪些期待?': 'ai_search_expectations',
    '您愿意使用可以信息提取/总结的搜索引擎吗': 'ai_search_feature_preference',
    '您是否认同AI搜索会使得搜索到的内容和价值更丰富': 'ai_search_value',
    '您是否认同AI搜索会更符合用户期待': 'ai_search_user_expectation'
})

# 将选项转换为数字
df.replace({
    'gender': {'男': 1, '女': 0},
    'age': {'小于18岁': 0, '18-26岁': 1, '大于26岁': 2},
    'education': {'高中/中专及以下': 0, '大学本科和专科': 1, '硕士及以上': 2},
    'chatbot_search_knowledge': {'没有且不想了解': 0, '没有但想了解': 1, '有些了解，尝试过': 2, '了解，经常用': 3},
    'occupation': {
        '在校生': 0,
        '互联网从业人员': 1,
        '私营职业者': 2,
        '政府工作人员': 3,
        '教师': 4,
        '其他': 5
    },
    'search_frequency': {
        '每天至少一次': 3,
        '每周至少一次': 2,
        '每两周至少一次': 1,
        '每月至少一次': 0,
        '几乎不用': -1
    },
    'traditional_search_efficiency': {
        '非常不认同': -2,
        '比较不认同': -1,
        '一般': 0,
        '比较认同': 1,
        '非常认同': 2
    },
    'satisfaction': {
        '非常不满意': -2,
        '比较不满意': -1,
        '一般': 0,
        '比较满意': 1,
        '非常满意': 2
    },
    'trust': {
        '非常不信任': -2,
        '比较不信任': -1,
        '一般': 0,
        '比较信任': 1,
        '非常信任': 2
    },
    'user_friendly': {
        '非常不重要': -2,
        '比较不重要': -1,
        '一般': 0,
        '比较重要': 1,
        '非常重要': 2
    },
    'accuracy': {
        '非常不重要': -2,
        '比较不重要': -1,
        '一般': 0,
        '比较重要': 1,
        '非常重要': 2
    },
    'aesthetics': {
        '非常不重要': -2,
        '比较不重要': -1,
        '一般': 0,
        '比较重要': 1,
        '非常重要': 2
    },
    'chatbot_search_preference': {
        '非常不愿意': -2,
        '比较不愿意': -1,
        '一般': 0,
        '比较愿意': 1,
        '非常愿意': 2
    },
    'ai_search_feature_preference': {
        '非常不愿意': -2,
        '比较不愿意': -1,
        '一般': 0,
        '比较愿意': 1,
        '非常愿意': 2
    },
    'ai_search_value': {
        '非常不认同': -2,
        '比较不认同': -1,
        '一般': 0,
        '比较认同': 1,
        '非常认同': 2
    },
    'ai_search_user_expectation': {
        '非常不认同': -2,
        '比较不认同': -1,
        '一般': 0,
        '比较认同': 1,
        '非常认同': 2
    }
}, inplace=True)

# 将多选题的选项转换为二进制
search_purpose_options = ['找资料', '看新闻', '听歌、看视频或娱乐相关', '查询作业答案', '其他']
for i, option in enumerate(search_purpose_options):
    df['search_purpose_' + str(i)] = df['search_purpose'].apply(lambda x: int(option in x))

ai_search_expectations_options = [
    '更智能的搜索模式',
    '更快速和高效的搜索结果',
    '更好的用户体验',
    '更广泛的覆盖范围',
    '更好地保护隐私和数据安全',
    '其他'
]
for i, option in enumerate(ai_search_expectations_options):
    df['ai_search_expectations_' + str(i)] = df['ai_search_expectations'].apply(lambda x: int(option in x))

# 删除原始列
df.drop(['search_purpose', 'ai_search_expectations'], axis=1, inplace=True)

# 将缺失值替换为0
df.fillna(0, inplace=True)

# 保存处理后的数据
df.to_csv('E:/13届市调分析大赛(研究生)/调查问卷数据处理后.csv', index=False)
