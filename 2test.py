'''
Author: whr1241 2735535199@qq.com
Date: 2022-05-05 16:15:02
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-23 17:26:41
FilePath: \InfraredPPG1.1\2test.py
Description: 对图像数据集进行处理得到原始数据，第三章使用
'''
import cv2
import dlib
import time
import numpy as np
import matplotlib.pyplot as plt

# 得到鼻子和脸颊rectangle等空间均值
def GetFourROI(frame1, detector1, predictor1):
    """
    根据特征点定位获取几个ROI的位置，展示一下，并得到五个ROI均值
    第一个是直接检测到的人脸矩形均值，后面4个如函数中定义额头、左脸颊、右脸颊、鼻子
    """
    
    rects = detector1(frame1, 0)  # 人脸数rects（耗时），必须是灰度化的图像呢
    rect = rects[0]

    if len(rects) == 1:
        face_points = predictor1(frame1, rects[0])  #
        points = np.zeros((len(face_points.parts()), 2))  # 将这些点存储在Numpy数组array中，这样我们就可以很容易地通过切片得到x和y的最小值和最大值68行2列
        for j, part in enumerate(face_points.parts()):  # numerate用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            points[j] = (part.x, part.y)  # 把特征点的X坐标和Y坐标存进points数组
        # 得到额头矩形的左右上下边界值
        min_x1 = int(points[21, 0])
        min_y1 = int(min(points[21, 1], points[22, 1]))
        max_x1 = int(points[22, 0])
        max_y1 = int(max(points[21, 1], points[22, 1]))
        left1 = min_x1  # 横坐标应该对应像素第几列
        right1 = max_x1
        top1 = min_y1 - (max_x1 - min_x1)  # y是第几行，高度
        bottom1 = max_y1
        # 得到左脸颊的ROI
        left2 = int(points[2, 0]+0.3*(points[30, 0]-points[2, 0]))  # 左边
        top2 = int(max(points[1, 1], points[28, 1]))
        right2 = int(points[2, 0]+0.6*(points[30, 0]-points[2, 0]))
        bottom2 = int(max(points[2, 1], points[30, 1]))
        # 得到右脸颊的ROI
        left3 = int(points[30, 0]+0.3*(points[14, 0]-points[30, 0])) 
        top3 = int(max(points[15, 1], points[28, 1]))
        right3 = int(points[30, 0]+0.6*(points[14, 0]-points[30, 0]))
        bottom3 = int(max(points[14, 1], points[30, 1]))
        # 鼻子ROI
        left4 = int(points[30, 0]-0.2*(points[30, 0]-points[2, 0])) 
        top4 = int(points[28, 1])
        right4 = int(points[30, 0]+0.2*(points[14, 0]-points[30, 0]))
        bottom4 = int(points[30, 1])

        show_frame1 = frame1.copy()
        cv2.rectangle(show_frame1, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 255, 0), 1)
        cv2.rectangle(show_frame1, (int(left1), int(top1)), (int(right1), int(bottom1)), (0, 255, 0), 1)
        cv2.rectangle(show_frame1, (int(left2), int(top2)), (int(right2), int(bottom2)), (0, 255, 0), 1)
        cv2.rectangle(show_frame1, (int(left3), int(top3)), (int(right3), int(bottom3)), (0, 255, 0), 1)
        cv2.rectangle(show_frame1, (int(left4), int(top4)), (int(right4), int(bottom4)), (0, 255, 0), 1)
        # 画出特征点
        i = 1
        for pt in face_points.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(show_frame1, pt_pos, 1, (0, 0, 255), 2)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(show_frame1, str(i), pt_pos, font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)
            i = i + 1

        # 得到5个均值
        frame = frame1[:, :, 0]

        FaceMean = np.mean(frame[rect.top():rect.bottom(), rect.left():rect.right()])
        ForeHeadMean = np.mean(frame[top1:bottom1, left1:right1])
        Cheek1Mean = np.mean(frame[top2:bottom2, left2:right2])
        Check2Mean = np.mean(frame[top3:bottom3, left3:right3])
        NoseMean = np.mean(frame[top4:bottom4, left4:right4])
        cv2.imshow('人脸和特征点', show_frame1)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # print([FaceMean, ForeHeadMean, Cheek1Mean, Check2Mean, NoseMean])
        Mean = [FaceMean, ForeHeadMean, Cheek1Mean, Check2Mean, NoseMean]
        return Mean   # 返回值是额头ROI的平均值和画出矩形框的图
    else:
        return [0, 0, 0, 0, 0]

    
if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()  # 人脸检测出矩形框
    predictor = dlib.shape_predictor('module\shape_predictor_81_face_landmarks.dat')

    video_path = r'I:\WHR\0-Dataset\DataBase\ir_heartrate_database\videos\17front'
    save_file_name = '17front'

    ROI_means = np.empty(shape=(5, 0))  # 定义一个5行0列的numpy来放数据
    print(ROI_means)
    n_frame = 5369  # 一共要人脸检测的张数
    # n_frame = 300  # 一共要人脸检测的张数
    start_index = 31 # 开始的索引号
    end_index = start_index+n_frame  # 结尾的索引号
    start_time = time.time()

    for i in range(start_index, end_index):  # 从31到5399
        frame = cv2.imread(video_path + r'\test{}.pgm'.format(i))  # 三通道
        print("Processing frame:", i, "/", n_frame+start_index-1)
        ROI_mean = GetFourROI(frame, detector, predictor)
        print(ROI_mean)
        ROI_mean1 = np.array(ROI_mean).reshape(5, 1)
        ROI_means = np.hstack((ROI_means, ROI_mean1))  # 在水平方向叠加，估计得到的数据shape是(5, 5369)
    end_time = time.time()
    print('用时：', end_time-start_time, '秒')
    raw_signal = ROI_means.tolist()   # 使用list形式容易保存为npy数据
    np.save("./Output/video_signal3/{}.npy".format(save_file_name), raw_signal)  # 以list保存

    # plot一下原始信号
    color_name = ['r', 'g', 'b', 'c', 'm']
    level_name = ['Rectangle', 'Forehead', 'LeftCheek', 'RightCheek', 'Nose']

    plt.figure(1)
    x = np.arange(0, ROI_means.shape[1])  # 返回一个有终点和起点的固定步长的排列做x轴
    for i in range(ROI_means.shape[0]):
        plt.plot(x, ROI_means[i, :], color=color_name[i], label=level_name[i])  # 绘制第i行,并贴出标签
    plt.legend()
    plt.title("original regions_mean")
    
    
    # plt.figure("ORIGINAL")
    # N = ROI_means.shape[0]
    # for n, ori in enumerate(ROI_means):
    #     plt.subplot(N,1,n+1)
    #     plt.plot(ori, 'g')
    #     # plt.title("ORIGINAL "+str(n))
    # plt.xlabel("Frames")
    # # plt.ylabel("Amplitude")

    plt.show()