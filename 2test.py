from ast import Pass
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import numpy as np
x = electrocardiogram()[2000:4000]
peaks, _ = find_peaks(x, distance=100)
# peaks, _ = find_peaks(x, height=0)
# peaks= find_peaks(x, distance=2)
# print('输出：', peaks)
# # print(type(peaks[0]))
# print(len(peaks[0]))
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()
