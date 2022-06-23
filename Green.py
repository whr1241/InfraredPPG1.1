'''
Author: whr1241 2735535199@qq.com
Date: 2022-06-15 21:15:29
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2022-06-17 20:33:11
FilePath: \InfraredPPG1.1\Green.py
Description: Green方法
'''
import time
import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
from biosppy.signals import ecg
import signal_tools as stools
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error

def get_ROI_mean(gray_frame, detector):
    """
    return:ROI均值和画出ROI的图像
    """
    rects = detector(gray_frame, 0)
    if len(rects) == 1:  # 确保只有一个人脸被检测到,或者返回0
        rect = rects[0]
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()

        show_frame = gray_frame.copy()  # 为什么要这样做复制一下？
        cv2.rectangle(show_frame, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 1)
        # ROI_mean = np.mean(gray_frame[int(top):int(bottom), int(left):int(right)])
        ROI_mean = np.mean(gray_frame[top:bottom, left:right])
        return ROI_mean, show_frame
    else:  # 没有检测到人脸该怎么办？
        return 0, 0



if __name__ == '__main__':

    # 图片数据集路径
    video_path = r'I:\DataBase\ir_heartrate_database\videos\06front'
    # video_path = r'I:\WHR\Dataset\1-Myself\5-haoran\video\subject1.1'
    detector = dlib.get_frontal_face_detector()  # 人脸检测出矩形框
    # 真值
    ecgdata = np.loadtxt(r"I:\DataBase\ir_heartrate_database\ecg\06\front_ecg.txt")
    # ecgdata = np.loadtxt(r"I:\WHR\Dataset\1-Myself\5-haoran\ecg\subject1.1.txt")
    # TODO:运行太慢，需要想办法优化
    # 设置全局参数
    start_index = 31
    end_index = 5399
    mean_gray = []

    j = start_index
    while j <= end_index:
        # 读取图片
        frame = cv2.imread(video_path + r'\{}.pgm'.format(j))  # 三通道
        print("Processing frame:", j, "/", end_index)
        # cv2.imshow("raw frame", frame)  
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 变成单通道
        gray_frame = frame[:, :, 1]  # 取绿色通道

        ROI_mean, showframe = get_ROI_mean(gray_frame, detector)
        mean_gray.append(ROI_mean)
        # cv2.imshow("ROI", showframe)  

        j = j + 1
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()  # 释放所有的资源
    
    # PPG信号处理
    Nor_mean_gray = stools.Z_ScoreNormalization(mean_gray)
    Fil_signal = stools.ButterworthFilter(Nor_mean_gray)
    PPG_signal = Fil_signal
    
    # ECG信号处理
    ecg_signal = ecgdata[:, 0]  # type? 应该是list
    ecg_signal = ecg_signal[1000*1:]
    out = ecg.ecg(ecg_signal, sampling_rate=1000., show=False)  # biosppy库功能 Tuple,应该是默认采样率1000
    times = out['heart_rate_ts']   # times是时间，长176
    bpm = out['heart_rate']  # 实时心率，对应时间的心率，长176

    # PPG与ECG结果对比
    # 实时心率计算
    win_i = 0  # 第几个窗口
    video_BPM = []
    real_BPM = []
    averageHR = 0
    win_start = 0
    win_end = 30*10  # 10s时间窗口
    realtime_win_start = 0
    realtime_win_end = 10
    while win_end < 5369:
        # averageHR = stools.fftTransfer1(data[win_start:win_end])  
        averageHR, averageHRs = stools.fftTransfer(PPG_signal[win_start:win_end], win_i) 

        print('最大五个：', averageHRs, '最大值：', averageHR)
        # 增加一个选择机制，看频域峰值哪个离上个最近
        # if len(video_BPM) > 0:
        #     # 找到离上个BPM值最近的一个
        #     averageHR = find_nearest(averageHRs, video_BPM[-1])
        #     # 防止突变,要满足非主导频率再进行判断
        #     if len(averageHRs) > 1 and abs(averageHR - video_BPM[-1]) > 15:
        #         averageHR = video_BPM[-1]
        #     print('第', win_i, '个时间窗口心率：', averageHR)
        #     print('')

        win_i = win_i + 1
        video_BPM.append(averageHR)

        real_average_heartrate_win = []
        for idx, tm in enumerate(times):
            if realtime_win_start < tm < realtime_win_end:
                real_average_heartrate_win.append(bpm[idx])
        real_BPM.append(np.mean(real_average_heartrate_win))  # 对前5/10秒ECG心率取平均

        # 步进为1s
        win_start += 30*1
        win_end += 30*1
        realtime_win_start += 1
        realtime_win_end += 1

    # 结果评价
    MSE = mean_squared_error(real_BPM, video_BPM)
    MAE = mean_absolute_error(real_BPM, video_BPM)
    RMSE = np.sqrt(MSE)  # 根均方误差(RMSE)
    MRE = max_error(real_BPM, video_BPM)  # 最大残差
    print("MSE", MSE)
    print("MAE", MAE)
    print("RMSE", RMSE)
    print("MRE", MRE)

    # 直观时频结果对比
    # 心率曲线一致性
    plt.figure("1")
    plt.title("Time-frequency domain comparison")
    plt.xlabel("time(sec)")
    plt.ylabel("heart rate(bpm)")
    plt.plot(video_BPM, label="iPPG")
    plt.plot(real_BPM, label="ECG")
    plt.legend()  # 展示每个数据对应的图像名称
    plt.show()