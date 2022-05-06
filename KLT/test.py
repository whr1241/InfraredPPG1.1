# 尝试手写一遍
import cv2
import numpy as np 
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation



if __name__ == '__main__':

    draw_bb = True  # draw bounding box
    save_to_file = True  #
    play_realtime = True  #

    cap = cv2.VideoCapture("testfile/14.mp4")
    n_frame = 5400
    frames = np.empty((n_frame,),dtype=np.ndarray)  # shape一维
    frames_draw = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)  # 放追踪的点

    for frame_idx in range(n_frame):
        _, frames[frame_idx] = cap.read()

    if draw_bb:
        n_object = int(input("Number of objects to track:"))  # 追踪的物体数量
        bboxs[0] = np.empty((n_object,4,2), dtype=float)
        for i in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Object %d"%(i),frames[0])  # 对第0帧进行选择ROI
            cv2.destroyWindow("Select Object %d"%(i))
            bboxs[0][i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
    else:
        n_object = 1
        bboxs[0] = np.array([[[291,187],[405,187],[291,267],[405,267]]]).astype(float)

    if save_to_file:
        out = cv2.VideoWriter('output/output14.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(frames[i].shape[1],frames[i].shape[0]))

    # 从第一帧开始，每两个连续帧做光流.
    startXs,startYs = getFeatures(cv2.cvtColor(frames[0],cv2.COLOR_RGB2GRAY),bboxs[0],use_shi=False)
    for i in range(1,n_frame):
        print('Processing Frame',i)
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i])
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1])
        
        # 更新坐标
        startXs = Xs
        startYs = Ys

        # 根据需要更新特征点
        n_features_left = np.sum(Xs!=-1)
        print('# of Features: %d'%n_features_left)
        if n_features_left < 15:
            print('Generate New Features')
            startXs,startYs = getFeatures(cv2.cvtColor(frames[i],cv2.COLOR_RGB2GRAY),bboxs[i])

        # 绘制边界框，可视化每个对象的特征点
        frames_draw[i] = frames[i].copy()
        for j in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i][j,:,:].astype(int))  # 用一个最小的矩形，把找到的形状包起来
            frames_draw[i] = cv2.rectangle(frames_draw[i], (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,0), 2)
            # for k in range(startXs.shape[0]):
            #     frames_draw[i] = cv2.circle(frames_draw[i], (int(startXs[k,j]),int(startYs[k,j])),3,(0,0,255),thickness=2)

        # 实时显示
        if play_realtime:
            cv2.imshow("win",frames_draw[i])
            cv2.waitKey(1)
        if save_to_file:
            out.write(frames_draw[i])
    
    if save_to_file:
        out.release()
    
    cap.release()