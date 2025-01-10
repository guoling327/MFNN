import matplotlib.pyplot as plt

#total_width, n = 0.3, 3
#width = total_width / n
width=0.20

name_list = ['Cora','Citeseer', 'Photo','Computers', 'Cornell', 'Chameleon', 'Wisconsin', 'Texas', 'Squirrel' ,'Actor']
num_list1 = [89.18,	80.90,	95.18,	91.05,	91.64,	70.72,	94.50,	93.61,	59.91,	41.07]
num_list2 = [87.73,	78.92,	85.36,	90.65,	84.26,	66.61,	91.50,	91.48,	37.10,	41.02]
num_list3 = [88.05,	79.90,	89.07,	90.71,	88.85,	64.62,	87.63,	92.13,	37.20,	40.10]

#num_list5 = [88.28,	80.35,	94.10,	68.75,	94.63,	92.79]

# 设置新的颜色列表
colors = ['#448ee4', '#fce166', '#e78ea5']

x = list(range(len(num_list1)))
plt.figure(figsize=(10,6))
plt.bar(x, num_list1, width=width, color=colors[0], label='Mixer.')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, color=colors[1],label='Mean.', tick_label=name_list)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width,color=colors[2],label='Sum.')
for i in range(len(x)):
    x[i] = x[i] + width


plt.legend()
plt.xticks(rotation=15)
plt.yticks()
plt.xlabel('Datasets')
plt.ylabel('Accuracy(%)')
plt.ylim(35,100)


plt.savefig('../node_classify/tu/fun3.jpg',dpi=600)
plt.savefig('../node_classify/tu/fun3.eps',dpi=600)
plt.show()



