'''
Author: whr1241 2735535199@qq.com
Date: 2022-04-19 14:34:56
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-20 00:58:56
FilePath: \InfraredPPG1.1\0test.py
Description: 对视频数据集进行处理
'''
import cv2
import dlib

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()  # 人脸检测出矩形框
    cap = cv2.VideoCapture(r'I:\WHR')  # 读取本地视频
    if not cap.isOpened():
        print("无法打开视频")
        exit()

    roi_avg_values = []  # 存放每一帧的ROI的均值
    times = []  # 存放时间？

    while True:
        ret, frame = cap.read()  # 读取视频中的一帧
        if not ret:  # 检查是否已到达视频结尾
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将帧转换为灰度图像，以便进行人脸检测

        faces = detector(gray)  # 使用Dlib检测人脸

        # 在帧上绘制人脸矩形
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 显示帧
        cv2.imshow('Video', frame)   
    # 释放视频和窗口
    cap.release()
    cv2.destroyAllWindows()
