import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.manifold import TSNE
import pylab as pl
pl.rcParams['font.sans-serif']=['SimHei']
pl.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")

def plot3D(datayz_true,datayz_true_result):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    tsne = TSNE(n_components=3) # TSNE dimensionality reduction
    normal = pd.DataFrame(tsne.fit_transform(datayz_true))  # Perform data dimensionality reduction and return results


    # PCA dimensionality reduction
    # PCAE = PCA(n_components=3)
    # normal = pd.DataFrame(PCAE.fit_transform(datayz_true))
    datayz_true_result = pd.DataFrame(datayz_true_result)


    ax = plt.subplot(111, projection='3d')  # Create a three-dimensional drawing project
    #  Draw the data points in three parts, differentiated in color
    colors = [  'black','red','yellow', 'blue', 'burlywood','blanchedalmond','gold','lightcoral','magenta','navy']
    for i in range(len(colors)):
        temp = normal[datayz_true_result[0] == i]
        x, y, z = temp[0], temp[1], temp[2]
        ax.scatter(x, y, z, c=colors[i])  # Plotting data points
    ax.set_zlabel('Z')  # Coordinate axes
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

# Read data and perform data
data = pd.read_excel("data.xlsx")
data_index = list(data)
data_index_chara = data_index[1:]
print(data)
print(data_index)

# Matching road class'
street_type = ['Rd','Dr', 'St','Ave','N','S','E','W','Blvd','Ln','Highway','Way','Pkwy','Hwy','Ct','SW','NE','Pl','NW','State',
               'Old','SE','Road','Cir','US','Creek','County','Hill','Park','Route','Lake','Trl','I','Valley','Ridge','Mill',
               'River','Oak','Pike','Loop']


# Matching Type - Description and Street - Note Solution to Solution Size Size - Use the way to convert all characters to lowercase matches
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

# Reset the index
data = data.reset_index(drop=True)
# Discrete value processing for a column that is essentially unique
disp_list = ['Description', 'Street', 'Side', 'Borough','Weather_Condition', 'Civil_Twilight','Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
             'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
disp_dict = {}
for i in disp_list:
    temp_dict = {}
    disp_set = list(set(data[i]))
    for j in range(len(disp_set)):
        temp_dict[disp_set[j]] = j
    disp_dict[i] = temp_dict

# Discretization by the above dictionary - i.e. replacement
for i in disp_list:
    temp_dict = disp_dict[i]
    # print(temp_dict)
    for k, v in temp_dict.items():
        # print(k,v)
        data[i] = data[i].replace(k, v)

# Fill all null values with zeros first
data.iloc[:,1:] = data.iloc[:,1:].fillna(0)
data.iloc[:,1:] = data.iloc[:,1:].replace(np.nan, 0)
# data.iloc[:,1:] = data.iloc[:,1:].replace(0, "")
# data.iloc[:,1:] = data.iloc[:,1:].replace(0, " ")
# data.iloc[:,1:] = data.iloc[:,1:].replace(0, "\t")
data.iloc[:,0] = data.iloc[:,0].fillna(3)

# Output processed overall data
data.to_excel("data_handle.xlsx")
print(data)
# Drawing
plot3D(np.array(data.iloc[:,1:]),np.array(data.iloc[:,0]))


#Divide training set and test set
data_select_2 = np.array(data)
import random
random.shuffle(data_select_2)
from sklearn.model_selection import train_test_split
#data: The data set that needs to be split
#random_state: Set the random seed to ensure that the same random number is generated every time you run
#test_size: The ratio of dividing the data into the training set 7:3
train_set, test_set = train_test_split(data_select_2, test_size=0.3, random_state=42)
train_set = pd.DataFrame(train_set)
test_set = pd.DataFrame(test_set)
train_set.columns = data_index
test_set.columns = data_index
train_set.to_excel("train.xlsx")
test_set.to_excel("test.xlsx")
