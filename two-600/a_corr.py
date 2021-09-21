# # Correlation study
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")
# Painting phase correlation coefficient thermogram
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

def test(df):
    dfData = df.corr()
    dfData.to_excel("corr.xlsx")
    plt.subplots(figsize=(9, 9)) # Set the size of the screen
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
    plt.savefig('corr.png')
    plt.show()

# Read training test data
train_set = pd.read_excel("data_handle.xlsx",index_col=0)

# df = df.drop(['Sub-category number'],axis =1)
print(train_set)
test(train_set)
