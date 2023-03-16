'''
Author: whr1241 2735535199@qq.com
Date: 2023-03-10 16:00:55
LastEditors: whr1241 2735535199@qq.com
LastEditTime: 2023-03-12 13:07:10
FilePath: \InfraredPPG1.1\3test.py
Description: Dlib人脸检测和特征点检测标注
'''
import cv2
import dlib


if __name__ == '__main__':
    pic = cv2.imread(r'I:\WHR\Figure\whr_blue3.jpg')
    pic = cv2.resize(pic, None, fx=0.6, fy=0.6)
    detector = dlib.get_frontal_face_detector()  # 人脸检测出矩形检测函数
    predictor = dlib.shape_predictor('module\shape_predictor_81_face_landmarks.dat')
    # gray_pic = pic[:, :, 1]  # 取绿色通道
    gray_pic = pic  # 取绿色通道
    rect = detector(gray_pic, 0)[0]
    left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
    show_pic = gray_pic.copy()  # 为什么要这样做复制一下？
    cv2.rectangle(show_pic, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 1)

    shape = predictor(gray_pic, rect)
    i = 1
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(show_pic, pt_pos, 1, (255, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(show_pic, str(i), pt_pos, font, 0.3, (0, 0, 255), 1,cv2.LINE_AA)
        i = i + 1
    
    cv2.imshow('人脸和特征点', show_pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()