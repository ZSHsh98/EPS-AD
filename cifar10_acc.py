import seaborn as sns
import matplotlib.pyplot as plt
from numpy import median
import pandas as pd
import numpy as np

# 背景蓝色
plt.switch_backend('agg')
plt.figure(figsize=(23,12))
plt.style.use('seaborn-darkgrid')

plt.rc('font',family='Liberation Serif')


font1 = {'family' : 'Liberation Serif',
'weight' : 'normal',
'size'   : 40,
}

font2 = {'family' : 'Liberation Serif',
'weight' : 'normal',
'size'   : 50,
}




sns.set(style="darkgrid")


data = {'acc':pd.Series([54.46,51.14,49.95,48.85,48.58,47.28,46.69,46.43,46.34,46.12,46.11]),
       'number':pd.Series(['Square','PGD','CW','A-DLR','A-CE','FAB','MM3','AA','MM5','T-AA','MM+'])}

df = pd.DataFrame(data)
print(df)

g=sns.barplot("number", y="acc", data=df,
            palette="hls")


for index, row in df.iterrows():
    print(row)
    g.text(row.name,row.acc, round(row.acc,2), color='black', ha="center",fontproperties = 'Liberation Serif', size = 35)

plt.ylabel('Accuracy (%)',font2)
plt.xlabel('Attacks',font2)
#plt.tick_params(labelsize=25)
plt.yticks(fontproperties = 'Liberation Serif', size = 40)
plt.xticks(fontproperties = 'Liberation Serif', size = 40)

plt.ylim(ymin = 43)
plt.ylim(ymax = 57)
plt.grid(linestyle='-.')

plt.savefig('cifar10_acc.png')
plt.savefig('cifar10_acc.pdf')