# -*- encoding: utf-8 -*-
'''
@file        : BVP_calculate.py
@time        : 2022/03/07 20:01:22
@author      : Lin-sudo
@description : 3Dlandmark[wholeface] + PSD
'''
import time
from cv2 import cvtColor
import face_alignment  # 需要python3.7版本的环境
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage import io
import collections
import cv2
import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mask(frame, fx, fy, pos="front"):
    """
    分割ROI
    :param frame:
    :param fx:
    :param fy:
    :param pos:
    :return:
    """
    mask = np.zeros(frame.shape, np.uint8)
    if pos == "front":
        # 正脸
        candidate_points = [0, 27, 16, 15, 14, 13, 12, 11, 5, 4, 3, 2, 1]
        poly_list = []
        for i in candidate_points:
            poly_list.append((int(fx[i]), int(fy[i])))  # list里元素是(int(fx[i]), int(fy[i]))
        poly_list = np.array(poly_list)  # shape?
        poly_list = poly_list.reshape((-1, 1, 2))  # 为什么要三维 ?
        mask = cv2.fillPoly(mask, [poly_list], 255)  # 填充凸多边形
    elif pos == "left45" or pos == "right45":
        candidate_points = [0, 27, 16, 15, 14, 13, 12, 11, 5, 4, 3, 2, 1]
        poly_list = []
        for i in candidate_points:
            poly_list.append((int(fx[i]), int(fy[i])))
        poly_list = np.array(poly_list)
        poly_list = poly_list.reshape((-1, 1, 2))
        mask = cv2.fillPoly(mask, [poly_list], 255)
    elif pos == "left90":
        candidate_points = [0, 1, 2, 3, 4, 30, 29, 28, 27]
        poly_list = []
        for i in candidate_points:
            poly_list.append((int(fx[i]), int(fy[i])))
        poly_list = np.array(poly_list)
        poly_list = poly_list.reshape((-1, 1, 2))
        mask = cv2.fillPoly(mask, [poly_list], 255)
    elif pos == "right90":
        candidate_points = [27, 28, 29, 30, 11, 12, 13, 14, 15]
        poly_list = []
        for i in candidate_points:
            poly_list.append((int(fx[i]), int(fy[i])))
        poly_list = np.array(poly_list)
        poly_list = poly_list.reshape((-1, 1, 2))
        mask = cv2.fillPoly(mask, [poly_list], 255)
    # 显示
    temp = cv2.polylines(frame, [poly_list], True, (255, 255, 255))  # 绘制多边形
    cv2.imshow("temp", temp)
    return mask



if __name__ == "__main__":

    # === 自定义 ===
    video_basepath = r'F:\1Study\0-Datasets\1-dataSet_tidy\1-Myself\5-haoran\test\whr_front'
    save_file_name = "whr_front"
    pos_mode = save_file_name[4:]  # 字符串"front"
    # pos_mode = save_file_name[2:]  # 字符串"front"
    # === END ===

    plot = True

    # sfd for SFD, dlib for Dlib and folder for existing bounding boxes.
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')

    # Optionally set detector and some additional detector parameters
    face_detector = 'sfd'
    face_detector_kwargs = {"filter_threshold": 0.8}

    # Run the 3D face alignment on a test image, without CUDA.
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True,
    #                                   face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True,
                                      face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

    # Initialize
    firstFrame = True
    iPPG = []
    start_time = time.time()
    # 视频总帧数
    for i in range(1, 5400):
        # 前1s曝光时间不算
        if i < 31:
            continue
        frame = cv2.imread(video_basepath+r'\{}.pgm'.format(i))  # 从31开始
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if firstFrame:
            firstFrame = False
            preds = fa.get_landmarks(frame)  # preds是什么
            mask = get_mask(frame, preds[0][:, 0], preds[0][:, 1], pos=pos_mode)  # 只检测第一帧的mask
            cv2.imshow("mask", mask)

        grayscale_level = cv2.mean(frame, mask=mask)  # 什么shape和type?
        iPPG.append(grayscale_level[0])

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)  # 在1ms内得到的键值
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

    end_time = time.time()
    duration = end_time - start_time
    print('使用时间：', duration)
    np.save("./Output/BVP_{}.npy".format(save_file_name), iPPG)

    # plot一下原始信号
    if plot:
        plt.title('original ppg signal')
        plt.plot(iPPG, 'b')
        plt.show()
