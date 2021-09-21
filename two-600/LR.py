#Classifier Comparison

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

# Call TSNE function for downscaling to 3D and then 3D visualization
def plot3D(datayz_true,datayz_true_result):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # TSNE downscaling
    tsne = TSNE(n_components=3) #TSNE downscaling
    normal = pd.DataFrame(tsne.fit_transform(datayz_true))  # Perform data dimensionality reduction and return results
    # PCA downscaling
    # PCAE = PCA(n_components=3)
    # normal = pd.DataFrame(PCAE.fit_transform(datayz_true))
    datayz_true_result = pd.DataFrame(datayz_true_result)


    ax = plt.subplot(111, projection='3d')  # Create a 3D drawing project
    #  Draw the data points in three parts, differentiated in color
    colors = [ 'red', 'yellow','blue','black',  'burlywood']
    for i in range(len(colors)):
        temp = normal[datayz_true_result[0] == i]
        x, y, z = temp[0], temp[1], temp[2]
        ax.scatter(x, y, z, c=colors[i])  # Plotting data points
    # ax.scatter(x[10:20], y[10:20], z[10:20], c='y')
    # ax.scatter(x[30:40], y[30:40], z[30:40], c='g')
    # x, y, z = abnormal[0], abnormal[1], abnormal[2]
    # #  Draw the data points in three parts, differentiated in color
    # ax.scatter(x, y, z, c='r')  # Plotting data points

    ax.set_zlabel('Z')  # Coordinate axes
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

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
# Normalization of data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# mm = MinMaxScaler()
mm = StandardScaler()
mm_data = mm.fit_transform(data)
# print(pd.DataFrame(mm_data))
X_train = mm.transform(X_train)
X_test = mm.transform(X_test)

#Disrupting the order of the data set
# index = [i for i in range(len(data))]
# random.shuffle(index)
# data = data[index]
# label = label[index]

# dataAll = pd.concat([pd.DataFrame(data),pd.DataFrame(label)],axis=1)




#Training and testing with models
estimator = LogisticRegression(penalty="l2",solver="liblinear",C=0.1,max_iter=1000)



estimator.fit(X_train,y_train)
pre = estimator.predict(X_test)

#Calculation accuracy
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("accuracy",accuracy_score(pre,y_test))
print("precision",precision_score(pre,y_test, average='macro'))
print("recall",recall_score(pre,y_test, average='macro'))
print( "f1_score", f1_score(pre,y_test,average='macro'))
# print( "auc", roc_auc_score(pre,y_test))

#Draw the fitting diagram
estimator2 = TSNE(n_components=2)  # TSNE downscaling
# estimator2 = PCA(n_components=2)
X_pca2 = estimator2.fit_transform(X_test)
X_pca = estimator2.fit_transform(X_train)
estimator.fit(X_pca,y_train)
y = np.array(pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], axis=0))
X_pca_All = np.array(pd.concat([pd.DataFrame(X_pca), pd.DataFrame(X_pca2)], axis=0))
y_predicted = estimator.predict(X_pca2)
# Draw a graph, draw a grid data distribution
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 绘制
N, M = 500, 500  # How many values are sampled in each direction
x1_min, x2_min = X_pca_All.min(axis=0)
x1_max, x2_max = X_pca_All.max(axis=0)
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)  # Generate grid sampling points
x_show = np.stack((x1.flat, x2.flat), axis=1)  # Test Points
y_predict = estimator.predict(x_show)
"""
'#FFFF00', '#76EE00'
'#FFAAAA', '#AAFFAA', '#AAAAFF'
"""
cm_light = mpl.colors.ListedColormap(['#FFFF00', '#76EE00', '#AAAAFF','#FFAAAA','#AAFFAA'])
cm_dark = mpl.colors.ListedColormap(['#FFFF00', '#76EE00', '#AAAAFF','#FFAAAA','#AAFFAA'])
plt.pcolormesh(x1, x2, y_predict.reshape(x1.shape), cmap=cm_light)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolors='k', s=50, cmap=cm_dark)  # Sample
# plt.scatter(X_pca2[:, 0], X_pca2[:, 1], s=120, facecolors='none', zorder=10)  # Sample test set



plt.xlabel(u'x', fontsize=13)
plt.ylabel(u'y', fontsize=13)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
plt.title(u'', fontsize=15)
plt.plot(color='green', label='流失')
# plt.grid()
plt.show()

