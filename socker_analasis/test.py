import cv2 as cv
capture=cv.VideoCapture(r'D:\Project\blog\socker_analasis\data\soccerout.avi')#调取摄像头视频
    #capture = cv.VideoCapture(‘视频路径’)
while(True):
    ret,frame=capture.read()#返回的是

    cv.imshow('video',frame)
    c=cv.waitKey(50)
    if c==27:
        break