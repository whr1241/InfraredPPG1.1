'''
Author: whr1241 2735535199@qq.com
Date: 2022-04-19 14:34:56
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-12 14:01:44
FilePath: \InfraredPPG1.1\0test.py
Description: 对视频数据集进行处理
'''
import cv2
import dlib



if __name__ == '__main__':
    frames = cv2.videocapture(0)
    if not frames.isOpened():
        print()