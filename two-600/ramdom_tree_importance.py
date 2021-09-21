#Use random forest to study the importance of features




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
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, ShuffleSplit  # Subset division method required for cross-validation

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

# Read official data
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

# Read training test data
train_set = pd.read_excel("train.xlsx",index_col=0)
test_set = pd.read_excel("test.xlsx",index_col=0)

# Process data until it can be fed into the model
y_train = list(train_set.iloc[:,0])
y_test = list(test_set.iloc[:,0])

X_train = np.array(train_set.iloc[:,1:])
X_test = np.array(test_set.iloc[:,1:])

data = np.concatenate((X_train,X_test),axis=0)
label =y_train.copy()
label.extend(y_test)
# Normalize the data
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

#Dimension importance
feature_label = list(train_set)[1:]
print(len(feature_label))
impotances = estimator.feature_importances_
result = pd.concat([pd.DataFrame(feature_label),pd.DataFrame(impotances)],axis=1)
result.to_excel("impotances_result.xlsx")

# #Draw histogram
#
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# Create a window with 8 x 6 dots and set the resolution to 80 pixels/inch
plt.figure(figsize=(8, 6), dpi=80)
# Create another subgraph with a size of 1 x 1
plt.subplot(1, 1, 1)
# Total number of columns
N = len(feature_label)
# A sequence containing the corresponding value of each bar
values = impotances
# A sequence containing the subscripts of each column
index = np.arange(N)
# The width of the column
width = 0.5
# Draw a histogram, the color of each column is violet
p2 = plt.bar(index, values, width, label="impotance", color="#87CEFA")
# Set horizontal axis label
plt.xlabel('feature')
# Set the vertical axis
plt.ylabel('importance')
plt.title('feature importance')
plt.xticks(index, feature_label,rotation=90)
plt.yticks(np.arange(0, 0.6,0.1))
plt.legend(loc="upper right")
plt.show()

# Generate the fitting results of the random forest
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
    # Use ipython terminal jupyter notebook to display.
    Image(graph.create_png())
    graph.write_png(filename)

