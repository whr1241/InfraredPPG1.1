#dlib+KLT+图片数据集+IOU
import cv2
import numpy as np 
import time
from KLT.getFeatures import getFeatures
from KLT.estimateAllTranslation import estimateAllTranslation
from KLT.applyGeometricTransformation import applyGeometricTransformation
import matplotlib.pyplot as plt
import dlib


# IOU计算函数
def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return: 
    """
    [[xmin1,ymin1],[xmax1,ymin1],[xmin1,ymax1],[xmax1,ymax1]] = box1[0,:,:]
    [[xmin2,ymin2],[xmax2,ymin2],[xmin2,ymax2],[xmax2,ymax2]] = box2[0,:,:]
    # xmin2, ymin2, xmax2, ymax2 = box2[0,:,:]
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积
 
    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou


# 真正的主功能函数
def objectTracking(video_path, play_realtime=False):
    # initilize
    global mean_gray
    # global 
    n_frame = 5369
    start_index = 31
    end_index = start_index+n_frame

    frames = np.empty((n_frame,),dtype=np.ndarray)
    frames_draw = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)  # 放四个角
    temp = np.empty((1,), dtype=np.ndarray)  # 定义空数组

    for i in range(start_index, end_index):
        frame = cv2.imread(video_path + r'\{}.pgm'.format(i))  # 三通道
        print("Processing frame:", i-start_index+1, "/", n_frame)
        frames[i-start_index] = np.array(frame)

    # 追踪的ROI个数
    n_object = 1
    bboxs[0] = np.empty((n_object,4,2), dtype=float)  # bboxs用来放矩形框四个角的坐标？
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
        bboxs[0][i, :, :] = np.array([[left1,top1],[right1,top1],[left1,bottom1],[right1,bottom1]]).astype(float)
        temp = bboxs[0]  # 取第一帧rectangle作为对比标准

    
    # 从第一帧开始，每两帧进行光流处理。
    startXs,startYs = getFeatures(cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY),bboxs[0],use_shi=False)
    for i in range(1,n_frame):
        print('Processing Frame',i)
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i])
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1])

        # update coordinates 更新坐标
        startXs = Xs
        startYs = Ys

        # update feature points as required 根据需要更新特性点
        n_features_left = np.sum(Xs!=-1)
        print('# of Features: %d'%n_features_left)
        if n_features_left < 15:
            print('Generate New Features')
            startXs,startYs = getFeatures(cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY),bboxs[i])

        # 绘制每个对象的边界框并可视化特征点
        frames_draw[i] = frames[i].copy()

        # # 判断rectangle是否需要更新
        iou = cal_iou(temp, bboxs[i])
        if iou < 0.99:
            temp = bboxs[i]

        for j in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(temp[j,:,:].astype(int))
            # (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i][j,:,:].astype(int))
            frames_draw[i] = cv2.rectangle(frames_draw[i], (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,0), 2)
            for k in range(startXs.shape[0]):
                frames_draw[i] = cv2.circle(frames_draw[i], (int(startXs[k,j]),int(startYs[k,j])),3,(0,0,255),thickness=2)


        show_frame, levels_mean, background = multiscale_value(xmin, ymin, boxw, boxh, frames[i])
        levels_mean = np.array(levels_mean).reshape(5, 1)
        mean_gray = np.hstack((mean_gray, levels_mean))  # 水平方向依次叠加
        backgrounds.append(background)
        cv2.imshow("ROI", show_frame)  # 展示frame并画出额头矩形，每帧只展示一瞬间
        cv2.imshow("win", frames_draw[i])

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()  # 释放所有的资源


def multiscale_value(xmin, ymin, boxw, boxh, frame):
    """
    得到多个尺度的均值
    """
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left = xmin
    right = xmin + boxw
    top = ymin
    bottom = ymin + boxh

    # 得到扩大后的人脸rectangle，l-1 level
    left_e = int(left - (right - left)*0.1)
    right_e = int(right + (right - left)*0.1)
    top_e = int(top - (bottom - top)*0.2)
    bottom_e = int(bottom + (bottom - top)*0.2)

    # l1 level的rectangle
    left1 = int(left + (right - left) * 0.2)
    right1 = int(right - (right - left) * 0.2)
    top1 = int(top + (bottom - top) * 0.1)
    bottom1 = int(bottom - (bottom - top) * 0.1)

    # l2 level的rectangle
    left2 = int(left + (right - left) * 0.3)
    right2 = int(right - (right - left) * 0.3)
    top2 = int(top + (bottom - top) * 0.3)
    bottom2 = int(bottom - (bottom - top) * 0.3)

    # l3 level的rectangle
    left3 = int(left + (right - left) * 0.4)
    right3 = int(right - (right - left) * 0.4)
    top3 = int(top + (bottom - top) * 0.4)
    bottom3 = int(bottom - (bottom - top) * 0.4)

    show_frame1 = frame.copy()  # 为什么要这样做复制一下？

    cv2.rectangle(show_frame1, (int(left_e), int(top_e)), (int(right_e), int(bottom_e)), (0, 0, 255), 1)  # 叠加绘制扩大化后的整个人脸rectangle
    cv2.rectangle(show_frame1, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)  # 叠加绘制额头roi矩形
    cv2.rectangle(show_frame1, (int(left1), int(top1)), (int(right1), int(bottom1)), (255, 0, 0), 1)  # 叠加绘制 l1 level 的rectangle
    cv2.rectangle(show_frame1, (int(left2), int(top2)), (int(right2), int(bottom2)), (255, 255, 0), 1)  # 叠加绘制 l2
    cv2.rectangle(show_frame1, (int(left3), int(top3)), (int(right3), int(bottom3)), (139, 0, 139), 1)  # 叠加绘制 l3

    level_e = frame1[int(top_e):int(bottom_e), int(left_e):int(right_e)]  # 扩大后的rectangle
    level_0 = frame1[int(top):int(bottom), int(left):int(right)]  #
    level_1 = frame1[int(top1):int(bottom1), int(left1):int(right1)]  #
    level_2 = frame1[int(top2):int(bottom2), int(left2):int(right2)]  #
    level_3 = frame1[int(top3):int(bottom3), int(left3):int(right3)]  #
    level_e_mean = np.mean(level_e)  #
    level_0_mean = np.mean(level_0)  #
    level_1_mean = np.mean(level_1)  #
    level_2_mean = np.mean(level_2)  #
    level_3_mean = np.mean(level_3)  #
    level_mean = [level_e_mean, level_0_mean, level_1_mean, level_2_mean, level_3_mean]  # list只能是一维的吧

    # 获得背景信号的平均值
    mask = np.ones_like(frame1, dtype=bool)  # 生成一个同shape的1填充的array
    mask[int(top_e):int(bottom_e), int(left_e):int(right_e)] = False  # mask部分填充0
    background = frame1[mask]  # 返回一维，从整张帧图中删去人脸rectangle的部分,长度上是对的
    background_mean = np.mean(background)  # 对背景区域求平均

    return show_frame1, level_mean, background_mean  # 返回show图、roi平均值、背景信号平均值



if __name__ == "__main__":

    mean_gray = np.empty(shape=(5, 0))
    backgrounds = []
    detector = dlib.get_frontal_face_detector()  # 人脸检测出矩形框

    # 数据集路径
    video_path = r'I:\DataBase\ir_heartrate_database\videos\02front'
    # video_path = r'I:\WHR\Dataset\1-Myself\2022.4.21\3heh\3heh_ppg\3.0'

    save_file_name = '02front'
    start_time = time.time()

    objectTracking(video_path, play_realtime=True)

    end_time = time.time()
    duration = end_time - start_time
    print('duration:', duration)
    raw_signal = mean_gray.tolist()  # 保存比人脸稍微小一些的尺度信号
    np.save("./Output/video_signal/BVP_smooth_{}.npy".format(save_file_name), raw_signal)

    # plot一下原始信号
    color_name = ['r', 'g', 'b', 'c', 'm']
    level_name = ['level_e_mean', 'level_0_mean', 'level_1_mean', 'level_2_mean', 'level_3_mean']

    # mean_gray = mean_gray[:, 101:601]
    plt.figure(1)
    x = np.arange(0, mean_gray.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
    for i in range(mean_gray.shape[0]):
        plt.plot(x, mean_gray[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    plt.plot(backgrounds, color='k', label='background')
    plt.legend()
    plt.title("original regions_mean")
    plt.show()