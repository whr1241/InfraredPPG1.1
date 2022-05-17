# -*- coding: utf-8 -*-
 
import numpy as np
import pywt
 
data = np.linspace(1, 4, 7)
print(data)
 
# pywt.threshold方法讲解：
#               pywt.threshold（data，value，mode ='soft'，substitute = 0 ）
#               data：数据集，value：阈值，mode：比较模式默认soft，substitute：替代值，默认0，float类型
 
#data:   [ 1.   1.5  2.   2.5  3.   3.5  4. ]
#output：[ 6.   6.   0.   0.5  1.   1.5  2. ]
#soft 因为data中1小于2，所以使用6替换，因为data中第二个1.5小于2也被替换，2不小于2所以使用当前值减去2，，2.5大于2，所以2.5-2=0.5.....
print ("---------------------soft:绝对值-------------------------")
print (pywt.threshold(data, 2, 'soft',6))
 
print ("---------------------hard:绝对值-------------------------")
 
#data:   [ 1.   1.5  2.   2.5  3.   3.5  4. ]
#hard data中绝对值小于阈值2的替换为6，大于2的不替换
print (pywt.threshold(data, 2, 'hard',6))
 
print ("---------------------greater-------------------------")
 
#data:   [ 1.   1.5  2.   2.5  3.   3.5  4. ]
#data中数值小于阈值的替换为6，大于等于的不替换
print (pywt.threshold(data, 2, 'greater',6))
print ("---------------------less-------------------------")
print (data)
#data:   [ 1.   1.5  2.   2.5  3.   3.5  4. ]
#data中数值大于阈值的，替换为6
print (pywt.threshold(data, 2, 'less',6))
