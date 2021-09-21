#用随机森林来研究特征的重要性


import numpy as np
import pandas as pd
import random

from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, ShuffleSplit  # 交叉验证所需的子集划分方法
import matplotlib as mpl
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")
# 读正式数据
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal
from sklearn import svm
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE
# import catboost as cb
import pylab as pl
pl.rcParams['font.sans-serif']=['SimHei']
pl.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")

# 读训练测试数据
train_set = pd.read_excel("train.xlsx",index_col=0)
test_set = pd.read_excel("test.xlsx",index_col=0)

# 处理数据到可以进模型
y_train = list(train_set.iloc[:,0])
y_test = list(test_set.iloc[:,0])

X_train = np.array(train_set.iloc[:,1:])
X_test = np.array(test_set.iloc[:,1:])

data = np.concatenate((X_train,X_test),axis=0)
label =y_train.copy()
label.extend(y_test)
# 对数据进行归一化标准化
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# mm = MinMaxScaler()
mm = StandardScaler()
mm_data = mm.fit_transform(data)
# print(pd.DataFrame(mm_data))
X_train = mm.transform(X_train)
X_test = mm.transform(X_test)

estimator = RandomForestClassifier(n_estimators=6, random_state=6)

estimator.fit(data,label)

#维度重要性
feature_label = list(train_set)[1:]
print(len(feature_label))
impotances = estimator.feature_importances_
result = pd.concat([pd.DataFrame(feature_label),pd.DataFrame(impotances)],axis=1)
result.to_excel("impotances_result.xlsx")

# #画柱状图
#
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(8, 6), dpi=80)
# 再创建一个规格为 1 x 1 的子图
plt.subplot(1, 1, 1)
# 柱子总数
N = len(feature_label)
# 包含每个柱子对应值的序列
values = impotances
# 包含每个柱子下标的序列
index = np.arange(N)
# 柱子的宽度
width = 0.5
# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index, values, width, label="impotance", color="#87CEFA")
# 设置横轴标签
plt.xlabel('feature')
# 设置纵轴标签
plt.ylabel('importance')
# 添加标题
plt.title('feature importance')
# 添加纵横轴的刻度
plt.xticks(index, feature_label,rotation=90)
plt.yticks(np.arange(0, 0.6,0.1))
# 添加图例
plt.legend(loc="upper right")
plt.show()

# 生成随机森林的拟合结果
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
from IPython.display import Image
from sklearn import tree
import pydotplus
import pylab as pl
pl.rcParams['font.sans-serif']=['SimHei']
pl.rcParams['axes.unicode_minus'] = False
import os
Estimators = estimator.estimators_
i = 0
for index, model in enumerate(Estimators):
    print(i)
    i = i+1
    filename = 'tree' + str(i) + '.png'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=feature_label,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 使用ipython的终端jupyter notebook显示。
    Image(graph.create_png())
    graph.write_png(filename)

