# 提取出心率
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib

# 得到整张人脸rectangle的均值
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

        left1 = int(left + (right - left) * 0.2)
        right1 = int(right - (right - left) * 0.2)
        top1 = int(top + (bottom - top) * 0.1)
        bottom1 = int(bottom - (bottom - top) * 0.2)

        face_mean = np.mean(frame1[top:bottom, left:right])
        cv2.rectangle(frame1, (int(left1), int(top1)), (int(right1), int(bottom1)), (255, 255, 0), 1)
        cv2.imshow("rectangle", frame1)  # 展示frame并画出额头矩形，每帧只展示一瞬间

        return face_mean
    else:
        return 0


# 2.3得到多尺度ROI和背景区域信号各自的平均值
def get_multiscale_value(frame1, detector):
    """
    输入
    :param frame1: 应该是灰度化后的frame
    :param detector:
    :param predictor:
    :return:
    show_frame1:
    level_mean:长度为5的list,包含了5个尺度的均值
    background_mean:背景信号均值
    bug：当rectangle值超出frame的时候求均值就会变为nan
    """
    rects = detector(frame1, 0)  # rectangles[[(323, 54) (410, 141)]]，是一个set还是dictionary？
    if len(rects) == 1:  # 确保只有一个人脸被检测到
        rect = rects[0]  # rects[0]应该是 [(323, 54) (410, 141)] class 'dlib.dlib.rectangle'

        # 得到检测到的人脸，l0 level 的rectangle
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()

        # 得到扩大后的人脸rectangle，l-1 level
        left_e = int(left - (right - left)*0.1)
        right_e = int(right + (right - left)*0.1)
        top_e = int(top - (bottom - top)*0.3)
        bottom_e = int(bottom + (bottom - top)*0.3)

        # l1 level的rectangle
        left1 = int(left + (right - left) * 0.2)
        right1 = int(right - (right - left) * 0.2)
        top1 = int(top + (bottom - top) * 0.2)
        bottom1 = int(bottom - (bottom - top) * 0.2)

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

        show_frame1 = frame1.copy()  # 为什么要这样做复制一下？

        cv2.rectangle(show_frame1, (int(left_e), int(top_e)), (int(right_e), int(bottom_e)), (255, 255, 0), 1)  # 叠加绘制扩大化后的整个人脸rectangle
        cv2.rectangle(show_frame1, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 1)  # 叠加绘制额头roi矩形
        cv2.rectangle(show_frame1, (int(left1), int(top1)), (int(right1), int(bottom1)), (255, 255, 0), 1)  # 叠加绘制 l1 level 的rectangle
        cv2.rectangle(show_frame1, (int(left2), int(top2)), (int(right2), int(bottom2)), (255, 255, 0), 1)  # 叠加绘制 l2
        cv2.rectangle(show_frame1, (int(left3), int(top3)), (int(right3), int(bottom3)), (255, 255, 0), 1)  # 叠加绘制 l3

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

    else:
        level_zero = [0, 0, 0, 0, 0]
        background_zero = 0
        return frame1, level_zero, background_zero



if __name__ == '__main__':
    video_path = r'F:\1Study\0-Datasets\1-dataSet_tidy\1-Myself\5-haoran\test\whr_front'
    detector = dlib.get_frontal_face_detector()  # 人脸检测出矩形框
    # 设置全局参数
    start_index = 1
    end_index = 5399
    i = start_index
    mean_gray = np.empty(shape=(5, 0))
    backgrounds = []
    start_time = time.time()

    while start_index <= i <= end_index:
        # frame = io.imread(img_list[i])  # 几通道？
        frame = cv2.imread(video_path + r'\{}.pgm'.format(i))  # 三通道
        print("Processing frame:", i, "/", end_index)
        cv2.imshow("frame", frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # mean_frame = get_face_rectangle_average(gray_frame, detector)
        show_frame, levels_mean, background = get_multiscale_value(gray_frame, detector)
        levels_mean = np.array(levels_mean).reshape(5, 1)
        mean_gray = np.hstack((mean_gray, levels_mean))  # 水平方向依次叠加
        backgrounds.append(background)
        cv2.imshow("ROI", show_frame)  # 展示frame并画出额头矩形，每帧只展示一瞬间

        i = i + 1
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()  # 释放所有的资源

    end_time = time.time()
    duration = end_time - start_time
    print('duration:', duration)

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

