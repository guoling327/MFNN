
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# data_order0 = np.load('homo_cora_0.npy')
data_order1 = np.load('homo_cora_1.npy')
data_order2 = np.load('homo_cora_2.npy')

# 将数据整理为列表形式，方便绘图
data = [data_order1, data_order2]

# 创建一个新的图形对象
plt.figure(figsize=(9, 7),dpi=600)
# 创建箱线图
#plt.boxplot([data1, data2], labels=['Order 0', 'Order 1'])

# 创建箱线图
boxprops = dict(linewidth=1.5, color='black')
flierprops = dict(marker='o', markerfacecolor='black', markersize=6)
medianprops = dict(linewidth=2, color='orange')
meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='red')
whiskerprops = dict(linewidth=1.5, color='black')
capprops = dict(linewidth=1.5, color='black')

bplot = plt.boxplot(data, patch_artist=True, labels=['1-simplices', '2-simplices'],
                    boxprops=boxprops, flierprops=flierprops, medianprops=medianprops,
                    meanprops=meanpointprops, whiskerprops=whiskerprops, capprops=capprops)

# 修改箱体的颜色
colors = ['skyblue', 'lightgreen']#, 'salmon'#'lightgreen'
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# 添加标题和标签
plt.gca().set_xticklabels(['1-simplices', '2-simplices'], fontsize=20)
# 添加标题和标签
plt.title('Cora',fontsize=25)
plt.ylabel('Homophily',fontsize=23)
plt.xlabel('Simplices',fontsize=23)
# 显示网格线以增强可读性
plt.grid(True)
plt.savefig(f'/home/luwei/MDSGNN/node_classify/tu/homocora11.jpg', dpi=600)
# 显示图表
plt.show()