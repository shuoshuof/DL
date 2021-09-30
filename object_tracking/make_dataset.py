import cv2
import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional
from cvzone.HandTrackingModule import HandDetector
all_data = []
all_label = []
numpy_data = []
numpy_label = []
def add_feature(feature,label):
    global all_data,all_label
    feature=np.array(feature)
    feature =feature[np.newaxis,:,:]
    all_data.append(feature)
    all_label.append(label)
    # print(all_data)
    # print(np.array(all_data).shape)
def main():
    global all_data,all_label
    if os.path.exists('./x_l.npy') :
        all_data  = list(np.load("x_l.npy"))
    if os.path.exists('./y_l.npy'):
        all_label = list(np.load("y_l.npy"))
        print("原数据大小：",len(all_data))
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    type1 = 0
    type2 = 0
    type3 = 0
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)  # With Draw
        # hands = detector.findHands(img, draw=False)  # No Draw
        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmarks points
            # print(np.array(lmList1).shape)
            bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h
            centerPoint1 = hand1["center"]  # center of the hand cx,cy
            handType1 = hand1["type"]  # Hand Type Left or Right

            # print(len(lmList1),lmList1)
            # print(bbox1)
            # print(centerPoint1)
            fingers1 = detector.fingersUp(hand1)
            #length, info, img = detector.findDistance(lmList1[8], lmList1[12], img) # with draw
            #length, info = detector.findDistance(lmList1[8], lmList1[12])  # no draw

            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmarks points
                bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
                centerPoint2 = hand2["center"]  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type Left or Right
                #手指是否伸直
                fingers2 = detector.fingersUp(hand2)
                # print(fingers1, fingers2)
                #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img) # with draw
                length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)  # with draw
        # cv2.putText(img, "FPS : " + str((fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        if  len(hands) == 1:
            if k == 27:
                all_data = np.array(all_data)
                all_label = np.array(all_label)
                break
            elif  k==49:
                add_feature(lmList1,0)
                type1+=1
                print("type1=",type1)
            elif k == 50:
                add_feature(lmList1,1)
                type2+=1
                print("type2=",type2)
            elif k == 51:
                add_feature(lmList1,2)
                type3+=1
                print("type3=",type3)
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    np.save("x_l", all_data)
    np.save("y_l", all_label)
    print("已保存")
    print(all_data)
    print(all_data.shape)
    print(all_label)
def make_dataset():
    global all_data,all_label
    if not (os.path.exists('./0_feature')):
        os.mkdir("./0_feature")
    if not (os.path.exists('./1_feature')):
        os.mkdir("./1_feature")
    if not (os.path.exists('./2_feature')):
        os.mkdir("./2_feature")

    detector = HandDetector(detectionCon=0.8, maxHands=1)
    for i in range(0,3):
        path = './%d'%i
        num = 0
        for f in os.listdir(path):
            extension = os.path.splitext(f)[-1]
            if (extension == '.jpg'):
                # img = cv.imread(os.path.join(path, f),cv.IMREAD_GRAYSCALE)
                img = cv2.imread(os.path.join(path, f))
                hands, img = detector.findHands(img)
                if hands:
                    hand1 = hands[0]
                    # print(hands[0])
                    lmList1 = hand1["lmList"]  # List of 21 Landmarks points
                    # print(np.array(lmList1).shape)
                    bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h
                    centerPoint1 = hand1["center"]  # center of the hand cx,cy
                    handType1 = hand1["type"]  # Hand Type Left or Right
                    path1= './%d_feature/'%i
                    path2 = '%d.jpg'%num
                    num+=1
                    cv2.imwrite(os.path.join(path1,path2), img)
                    if i == 0:
                        add_feature(lmList1, 0)
                    if i == 1:
                        add_feature(lmList1, 1)
                    if i == 2:
                        add_feature(lmList1, 2)
                # cv2.imwrite('')
                else:
                    cv2.imshow("Image", img)
                    k=cv2.waitKey(1)
                    if k==27:
                        continue
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    np.save("x", all_data)
    np.save("y", all_label)
    print(all_data)
    print(all_data.shape)
    print(all_label)
main()
# make_dataset()