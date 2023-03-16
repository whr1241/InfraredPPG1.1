'''
Author: whr1241 2735535199@qq.com
Date: 2022-06-11 20:37:03
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2022-06-21 16:16:21
FilePath: \InfraredPPG1.1\Video_Synthesis.py
Description: 用图片数据集合成视频
这个头文件对于之前的py文件也能生成头文件，只是速度比较慢
'''

import cv2

# # 将数据集合成视频
# pict_basepath = r'I:\DataBase\ir_heartrate_database\videos\17left45'
# savename = '17left45'  # 生成视频名称
# videoWrite = cv2.VideoWriter("output/{}.avi".format(savename),cv2.VideoWriter_fourcc('X','V','I','D'),30,(1280,720))
# #  文件名   ，  编码器   ，帧率   ，  图片大小
# for i in range(1,5399):
#     img = cv2.imread(pict_basepath+r'\test{}.pgm'.format(i))
#     cv2.imshow('frame',img)
#     cv2.waitKey(1)
#     videoWrite.write(img)
# videoWrite.release()
# cv2.destroyAllWindows()
# print("end!")


# 将频域图合称为视频
pict_basepath = r'output\picture_video'
savename = '11front_F'  # 生成视频名称
videoWrite = cv2.VideoWriter("output/{}.avi".format(savename),cv2.VideoWriter_fourcc('X','V','I','D'),2,(1000,800))
#  文件名   ，  编码器   ，帧率   ，  图片大小
for i in range(105,169):
    img = cv2.imread(pict_basepath+r'\{}.png'.format(i))
    cv2.imshow('frame',img)
    cv2.waitKey(1)
    videoWrite.write(img)
videoWrite.release()
cv2.destroyAllWindows()
print("end!")