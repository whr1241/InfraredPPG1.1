#!/usr/bin/env python
# -*- coding  : utf-8 -*-
# @Time       : 2022/3/27 16:54
# @Author     : wanghaoran
# @Site       : SCNU
# @File       : Realtimedetect1.0.py
# @Description: 近红外相机+逐帧检测+实时人脸rectangle平均
# @Software   : PyCharm
import cv2
import dlib
import pyrealsense2 as rs
import numpy as np
import time

"""
可以试试低分辨率，再试试高分辨率
低分辨率三分钟用时180.5s，framerate: 29.9|十分钟604s
高分辨率三分钟用时425s，framerate: 12.7
"""


# 2.0.0得到整张人脸rectangle的均值
def get_face_rectangle_average(frame1, detector1):
    """
    :param frame1: 输入的二维灰度图片
    :param detector1: dlib函数
    :return:
    """
    rects = detector1(frame1, 0)  # 人脸数rects（耗时），必须是灰度化的图像呢
    if len(rects) == 1:  # 确保只有一个人脸被检测到
        rect = rects[0]

        # 得到检测到的人脸，l0 level 的rectangle
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
        face_mean = np.mean(frame1[top:bottom, left:right])
        return face_mean
    else:
        return 0


if __name__ == '__main__':

    detector = dlib.get_frontal_face_detector()  # 人脸检测出矩形框

    # 配置深度流
    pipeline = rs.pipeline()  # 有个未解决的提醒
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # ?

    # 开始推流
    pipeline_profile = pipeline.start(config)

    # 打开/关闭IR
    RSdevice = pipeline_profile.get_device()
    depth_sensor = RSdevice.query_sensors()[0]
    emitter = depth_sensor.get_option(rs.option.emitter_enabled)
    depth_sensor.set_option(rs.option.emitter_enabled, False)  # 应该是关闭补光吧

    start_time = time.time()
    all_frames = 0
    mean_infrared = []

    while all_frames <= 30*60*30:
        frames = pipeline.wait_for_frames()
        ir_frame_left = frames.get_infrared_frame(1)
        ir_left_image = np.asanyarray(ir_frame_left.get_data())  # 将图像转换为ndarray(480, 640),二维
        cv2.imshow('ir_left_image', ir_left_image)
        mean_frame = get_face_rectangle_average(ir_left_image, detector)
        mean_infrared.append(mean_frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            pipeline.stop()
            cv2.destroyAllWindows()
            break
        all_frames = all_frames + 1

    end_time = time.time()
    duration = end_time - start_time



    print('duration:', duration)
    print('all_frames:', all_frames)
    print('framerate:', all_frames/duration)
    print('Timing signal:', mean_infrared)

    # 新建一个字典用来存储姓名、时序信号、用时、实际帧率
    name = 'whr'
    save_signal = {'name': name, 'signal': mean_infrared, 'time': duration, 'framerate': all_frames/duration}
    # np.save("./Output/subject_test.npy", save_signal)
