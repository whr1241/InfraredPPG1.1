# dlib+KLT+图片数据集+IOU
import cv2
import numpy as np
import time
from KLT.getFeatures import getFeatures
from KLT.estimateAllTranslation import estimateAllTranslation
from KLT.applyGeometricTransformation import applyGeometricTransformation
import matplotlib.pyplot as plt
import dlib
import signal_tools as stools
import video_tools as vtools


# 真正的主功能函数
def objectTracking(video_path, play_realtime=False):
    # initilize
    global mean_gray
    # global
    n_frame = 5369
    start_index = 31
    end_index = start_index + n_frame

    frames = np.empty((n_frame,), dtype=np.ndarray)
    frames_draw = np.empty((n_frame,), dtype=np.ndarray)
    bboxs = np.empty((n_frame,), dtype=np.ndarray)  # 放四个角
    temp = np.empty((1,), dtype=np.ndarray)  # 定义空数组

    for i in range(start_index, end_index):
        frame = cv2.imread(video_path + r'\{}.pgm'.format(i))  # 三通道
        print("Processing frame:", i - start_index + 1, "/", n_frame)
        frames[i - start_index] = np.array(frame)

    # 追踪的ROI个数
    n_object = 1
    bboxs[0] = np.empty((n_object, 4, 2), dtype=float)  # bboxs用来放矩形框四个角的坐标？
    for i in range(n_object):
        # cv2.imshow('frame', frames[0])
        # cv2.waitKey(10000)
        faces = detector(frames[0])  # dlib需要三通道
        rect = faces[0]
        left1, right1, top1, bottom1 = rect.left(), rect.right(), rect.top(), rect.bottom()
        # left1 = int(left + (right - left) * 0.2)
        # right1 = int(right - (right - left) * 0.2)
        # top1 = int(top)
        # bottom1 = int(bottom - (bottom - top) * 0.2)
        bboxs[0][i, :, :] = np.array([[left1, top1], [right1, top1], [left1, bottom1], [right1, bottom1]]).astype(float)
        temp = bboxs[0]  # 取第一帧rectangle作为对比标准

    # 从第一帧开始，每两帧进行光流处理。
    startXs, startYs = getFeatures(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), bboxs[0], use_shi=False)
    for i in range(1, n_frame):
        print('Processing Frame', i)
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i - 1], frames[i])
        Xs, Ys, bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i - 1])

        # update coordinates 更新坐标
        startXs = Xs
        startYs = Ys

        # update feature points as required 根据需要更新特性点
        n_features_left = np.sum(Xs != -1)
        print('# of Features: %d' % n_features_left)
        if n_features_left < 15:
            print('Generate New Features')
            startXs, startYs = getFeatures(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), bboxs[i])

        # 绘制每个对象的边界框并可视化特征点
        frames_draw[i] = frames[i].copy()

        # # 判断rectangle是否需要更新
        iou = stools.cal_iou(temp, bboxs[i])
        if iou < 0.99:
            temp = bboxs[i]

        for j in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(temp[j, :, :].astype(int))
            # (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i][j,:,:].astype(int))
            frames_draw[i] = cv2.rectangle(frames_draw[i], (xmin, ymin), (xmin + boxw, ymin + boxh), (255, 0, 0), 2)
            for k in range(startXs.shape[0]):
                frames_draw[i] = cv2.circle(frames_draw[i], (int(startXs[k, j]), int(startYs[k, j])), 3, (0, 0, 255),
                                            thickness=2)

        show_frame, levels_mean = vtools.gridding_value(xmin, ymin, boxw, boxh, frames[i])
        levels_mean = np.array(levels_mean).reshape(25, 1)
        mean_gray = np.hstack((mean_gray, levels_mean))  # 水平方向依次叠加
        cv2.imshow("ROI", show_frame)  # 展示frame并画出额头矩形，每帧只展示一瞬间
        cv2.imshow("win", frames_draw[i])

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()  # 释放所有的资源



if __name__ == "__main__":

    mean_gray = np.empty(shape=(25, 0))
    backgrounds = []
    detector = dlib.get_frontal_face_detector()  # 人脸检测出矩形框

    # 数据集路径
    video_path = r'I:\DataBase\ir_heartrate_database\videos\06front'
    # video_path = r'I:\WHR\Dataset\1-Myself\2022.4.21\3heh\3heh_ppg\3.4'

    save_file_name = '06front'
    # save_file_name = 'heh3.4'
    start_time = time.time()

    objectTracking(video_path, play_realtime=True)

    end_time = time.time()
    duration = end_time - start_time
    print('duration:', duration)
    raw_signal = mean_gray.tolist()  # 保存比人脸稍微小一些的尺度信号
    np.save("./Output/video_signal/BVP_grid_{}.npy".format(save_file_name), raw_signal)  # 保存list信号

    # mean_gray = mean_gray[:, 101:601]
    plt.figure('original grid signals')
    x = np.arange(0, mean_gray.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
    for i in range(mean_gray.shape[0]):
        plt.plot(x, mean_gray[i, :])  # 绘制第i行,并贴出标签
    plt.title("original regions_mean")
    plt.show()
