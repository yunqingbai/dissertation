import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.manifold import TSNE
import pylab as pl
pl.rcParams['font.sans-serif']=['SimHei']
pl.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")
# 调用TSNE降维到三维，然后三维可视化的函数
def plot3D(datayz_true,datayz_true_result):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # TSNE降维
    tsne = TSNE(n_components=3) # TSNE降维
    normal = pd.DataFrame(tsne.fit_transform(datayz_true))  # 进行数据降维,并返回结果

    # PCA降维
    # PCAE = PCA(n_components=3)
    # normal = pd.DataFrame(PCAE.fit_transform(datayz_true))
    datayz_true_result = pd.DataFrame(datayz_true_result)


    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    colors = [  'black','red','yellow', 'blue', 'burlywood','blanchedalmond','gold','lightcoral','magenta','navy']
    for i in range(len(colors)):
        temp = normal[datayz_true_result[0] == i]
        x, y, z = temp[0], temp[1], temp[2]
        ax.scatter(x, y, z, c=colors[i])  # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

# 读数据并进行数据处理
data = pd.read_excel("data.xlsx")
data_index = list(data)
data_index_chara = data_index[1:]
print(data)
print(data_index)

# 匹配道路类'
street_type = ['Rd','Dr', 'St','Ave','N','S','E','W','Blvd','Ln','Highway','Way','Pkwy','Hwy','Ct','SW','NE','Pl','NW','State',
               'Old','SE','Road','Cir','US','Creek','County','Hill','Park','Route','Lake','Trl','I','Valley','Ridge','Mill',
               'River','Oak','Pike','Loop']

# 匹配类型--Description和Street -- 注意解决上面的关键词的大小写的方案--采用将所有字符都转换为小写进行匹配的方式
for i in range(len(data['Description'])):
    temp = data['Description'][i].lower().strip().split()
    data['Description'][i] = 0
    for j in street_type:
        if j.lower() in temp:
            data['Description'][i] = j
            break
for i in range(len(data['Street'])):
    temp = data['Street'][i].lower().strip().split()
    data['Street'][i] = 0
    for j in street_type:
        if j.lower() in temp:
            data['Street'][i] = j
            break

# 重置index
data = data.reset_index(drop=True)
# 对基本唯一的某列列进行离散值处理
disp_list = ['Description', 'Street', 'Side', 'Borough','Weather_Condition', 'Civil_Twilight','Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
             'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
disp_dict = {}
for i in disp_list:
    temp_dict = {}
    disp_set = list(set(data[i]))
    for j in range(len(disp_set)):
        temp_dict[disp_set[j]] = j
    disp_dict[i] = temp_dict

# 通过上述的字典进行离散化处理-即替换
for i in disp_list:
    temp_dict = disp_dict[i]
    # print(temp_dict)
    for k, v in temp_dict.items():
        # print(k,v)
        data[i] = data[i].replace(k, v)

# 将所有空值先用0填充
data.iloc[:,1:] = data.iloc[:,1:].fillna(0)
data.iloc[:,1:] = data.iloc[:,1:].replace(np.nan, 0)
# data.iloc[:,1:] = data.iloc[:,1:].replace(0, "")
# data.iloc[:,1:] = data.iloc[:,1:].replace(0, " ")
# data.iloc[:,1:] = data.iloc[:,1:].replace(0, "\t")
data.iloc[:,0] = data.iloc[:,0].fillna(3)

# 输出处理后的整体数据
data.to_excel("data_handle.xlsx")
print(data)
# 画图
plot3D(np.array(data.iloc[:,1:]),np.array(data.iloc[:,0]))


# 划分训练集和测试集
data_select_2 = np.array(data)
import random
random.shuffle(data_select_2)
from sklearn.model_selection import train_test_split
#data:需要进行分割的数据集
#random_state:设置随机种子，保证每次运行生成相同的随机数
#test_size:将数据分割成训练集的比例7：3
train_set, test_set = train_test_split(data_select_2, test_size=0.3, random_state=42)
train_set = pd.DataFrame(train_set)
test_set = pd.DataFrame(test_set)
train_set.columns = data_index
test_set.columns = data_index
train_set.to_excel("train.xlsx")
test_set.to_excel("test.xlsx")
