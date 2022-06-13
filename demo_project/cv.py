import cv2
import numpy as np


def init():
    # Provide points from Camera
    pts_src106 = np.array([[100,435],[420,475],[600,470],[345,255],[320,150]])
    pts_src113 = np.array([[585,475],[420,470],[280,475],[280,250],[280,175]])
    pts_src115 = np.array([[600,290],[320,290],[30,290],[300,350],[300,450]])
    # pts_src116 = np.array([[80,360],[400,375],[580,400],[525,475],[520,320]])
    pts_src116 = np.array([[65,360],[65,440],[480,410],[620,360],[620,440]])
    pts_src117 = np.array([[70,320],[260,330],[440,350],[440,475],[620,475]])
    pts_src118 = np.array([[160,475],[620,475],[400,285],[330,240],[160,225]])


    # Corresponding points in 2D Projection
    pts_dst106 = np.array([[353,243],[272,217],[288,115],[397,156],[486,170]])
    pts_dst113 = np.array([[289,371],[289,444],[290,506],[151,444],[53,444]])
    pts_dst115 = np.array([[326,510],[374,510],[450,510],[390,443],[390,388]])
    # pts_dst116 = np.array([[386,203],[480,157],[563,182],[447,265],[554,113]])
    pts_dst116 = np.array([[320,120],[320,215],[541,167],[480,120],[480,215]])
    pts_dst117 = np.array([[70,235],[160,255],[212,300],[214,390],[271,390]])
    pts_dst118 = np.array([[150,417],[20,417],[78,295],[62,199],[150,212]])


    # Calculate matrix H
    h106, status106 = cv2.findHomography(pts_src106, pts_dst106)
    h113, status113 = cv2.findHomography(pts_src113, pts_dst113)
    h115, status115 = cv2.findHomography(pts_src115, pts_dst115)
    h116, status116 = cv2.findHomography(pts_src116, pts_dst116)
    h117, status117 = cv2.findHomography(pts_src117, pts_dst117)
    h118, status118 = cv2.findHomography(pts_src118, pts_dst118)


    return [h106, h113, h115, h116, h117, h118]

def transform(h, initPoints, posX_index, posY_index):
    # provide a point you wish to map from image 1 to image 2
    # a = np.array([[154, 174]], dtype='float32')
   

    for i in range(len(initPoints)):

        pointToMap = np.array([[initPoints[i][posX_index], initPoints[i][posY_index]]], dtype='float32')
        pointToMap = np.array([pointToMap])
        # print("pointToMap is {} h is {}".format(pointToMap,h))

        # Finally, get the mapping
        # print("POints out", cv2.perspectiveTransform(pointToMap, h))
        if initPoints[i][2] == 106:
            camera_num = 0
        elif initPoints[i][2] == 113:
            camera_num = 1
        elif initPoints[i][2] == 115:
            camera_num = 2
        elif initPoints[i][2] == 116:
            camera_num = 3
        elif initPoints[i][2] == 117:
            camera_num = 4
        elif initPoints[i][2] == 118:
            camera_num = 5

        print("Camera is {}, Num is {}".format(initPoints[i][2],camera_num))

        initPoints[i][posX_index] = int(round(cv2.perspectiveTransform(pointToMap, h[camera_num])[0][0][0]))
        initPoints[i][posY_index] = int(round(cv2.perspectiveTransform(pointToMap, h[camera_num])[0][0][1]))

    # print("final points =  {}".format(initPoints))

    return initPoints