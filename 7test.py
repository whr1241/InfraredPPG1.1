'''
Author: whr1241 2735535199@qq.com
Date: 2023-03-20 14:31:42
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-22 10:17:16
FilePath: \InfraredPPG1.1\7test.py
Description: 将各个评价指标画出箱线图
'''

import matplotlib.pyplot as plt
import numpy as np

# Create sample data
np.random.seed(1)
data = np.random.normal(size=(100, 5))
# print(data)

all_data=[np.random.normal(0,std,100) for std in range(1,4)]
print(all_data)

# Create box plot
fig, ax = plt.subplots()
ax.boxplot(data)  # 对列数据画箱线图

# Set labels and title
ax.set_xlabel('Variable')
ax.set_ylabel('Value')
ax.set_title('Box plot')

# Show plot
plt.show()