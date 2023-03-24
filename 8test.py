import numpy as np
import signal_tools as stools
import matplotlib.pyplot as plt

data = np.load(r"output\video_signal3\01front.npy")
Plot = True
# show 原始时间数据
stools.show_signal(data, Plot)
plt.show()
print('hello')