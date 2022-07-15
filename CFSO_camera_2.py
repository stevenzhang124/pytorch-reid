import cv2

#print("Before URL")
#url = 'rtsp://admin:P@ssw0rd1234@10.15.56.62/media/video3'
url = 'rtsp://admin:pass@10.15.16.22/stream0'
# url = "rtsp://admin:edge1234@192.168.1." + str(117) + ":554/cam/realmonitor?channel=1&subtype=0"
# url = 'rtsp://admin:pass@10.15.40.11/stream0'
# url = '/home/edge/Downloads/20211228155627.ts'
cap = cv2.VideoCapture(url)
#print("After URL")

# fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
# fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
# width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
# out = cv2.VideoWriter('result_2.mp4', fourcc, fps, (width, height))  # 写入视频


while True:

    #print('About to start the Read command')
    ret, frame = cap.read()
    #print('About to show frame of Video.')
    # out.write(frame)  # 写入帧
    cv2.imshow("Capturing",frame)
    #print('Running..')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

#video1 1920x1080
#video2 1280x720
#video3 352x288