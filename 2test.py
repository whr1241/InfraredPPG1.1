import cv2

video_path = r'I:\WHR\Dataset\1-Myself\3-qinglin\videos\01front'
frame = cv2.imread(video_path + r'\{}.pgm'.format(1))  # 三通道
print(frame)
gray_frame = frame[:, :, 1]  # 取绿色通道
print(gray_frame)