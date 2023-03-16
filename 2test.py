'''
Author: whr1241 2735535199@qq.com
Date: 2022-05-05 16:15:02
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-10 16:06:41
FilePath: \InfraredPPG1.1\2test.py
Description: 对单个图像参数进行查看
'''

import cv2

video_path = r'I:\WHR\Dataset\1-Myself\3-qinglin\videos\01front'
frame = cv2.imread(video_path + r'\{}.pgm'.format(1))  # 三通道
print(frame)
print('end')
gray_frame = frame[:, :, 1]  # 取绿色通道
print(gray_frame)
