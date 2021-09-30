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
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(42, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, 128)
        self.out = nn.Linear(128, 3)

    def forward(self, x):
        x = functional.relu(self.hidden1(x))
        x = functional.relu(self.hidden2(x))
        x = functional.relu(self.hidden3(x))
        x = self.out(x)
        return x
def add_feature(feature,label):
    global all_data,all_label
    feature=np.array(feature)
    feature =feature[np.newaxis,:,:]
    all_data.append(feature)
    all_label.append(label)
    # print(all_data)
    # print(np.array(all_data).shape)
def move_rectangle(hand_x,hand_y,x,y,r,catch):
    r =int( (0.9**(r*0.1))*r)
    if (hand_x in range(x-r,x+r) and hand_y in range(y-r,y+r)) or catch:
        return hand_x,hand_y ,(0, 0, 255),1
    else:
        return x,y,(0, 255, 0),0
def scaliing(cx,cy,x,y,r,d):
    if cx in range(x-r,x+r) and cy in range(y-r,y+r):
        return x,y,int(d*0.2),(255,0, 0 )
    else:
        return x, y, r, (0, 255, 0)
def take_pic():
    global all_data,all_label

    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    # net = torch.load('./model.pth',map_location='cpu')
    net = MLP()
    net.load_state_dict(torch.load("hand_control_model.pth"))
    net.eval()
    x, y = 300, 300
    r = 50
    color = (0, 255, 0)
    catch = 0
    with torch.no_grad():
        while True:
            success, img = cap.read()
            hands, img = detector.findHands(img)  # With Draw
            # hands = detector.findHands(img, draw=False)  # No Draw
            if hands:
                # Hand 1
                hand1 = hands[0]
                lmList1 = hand1["lmList"]  # List of 21 Landmarks points
                print(np.array(lmList1).shape)
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
            left_top = (x-r,y-r)
            right_bottom = (x+r,y+r)
            cv2.rectangle(img,left_top,right_bottom,color,10)
            # print(img.shape)
            img = cv2.flip(img,1,dst=None)
            cv2.imshow("Image", img)
            k = cv2.waitKey(1)
            if k == 27:
                break
            if  len(hands) == 1:
                name  = ['抓住','包','ok']
                # lmList1 = lmList1.reshape((-1,42))
                lmList1 = torch.tensor(lmList1,dtype=torch.float32)
                lmList1 = lmList1.view(1,1,42)

                output = net(lmList1)
                print(output[0][0])
                result = output[0][0]
                confident  = functional.softmax(result)
                confident = confident[result.argmax()].numpy()
                print(name[result.argmax()],confident)
                if result.argmax() == 0:
                    hand_x,hand_y = centerPoint1
                    print(hand_x,hand_y)
                    x,y,color,catch = move_rectangle(hand_x,hand_y,x,y,r,catch)
                else:
                    catch =0
            elif len(hands) == 2:
                name  = ['抓住','缩放','ok']
                # lmList1 = lmList1.reshape((-1,42))
                lmList1 = torch.tensor(lmList1,dtype=torch.float32)
                lmList1 = lmList1.view(1,1,42)
                output1 = net(lmList1)
                result1 = output1[0][0]
                confident1 = functional.softmax(result1)
                confident1 = confident1[result1.argmax()].numpy()
                # print(name[output1.argmax()],confident1)

                lmList2 = torch.tensor(lmList2, dtype=torch.float32)
                lmList2 = lmList2.view(1, 1, 42)
                output2 = net(lmList2)
                result2 = output2[0][0]
                confident2 = functional.softmax(result2)
                confident2 = confident2[result2.argmax()].numpy()
                # print(name[output2.argmax()], confident2)
                x1, y1 = centerPoint1
                x2, y2 = centerPoint2
                d = ((x1-x2)**2 + (y1-y2)**2)**0.5
                cx,cy = (x1+x2)//2 , (y1+y2)//2
                if result1.argmax() == 1 and result2.argmax() == 1 and d >=30:
                    x,y,r,color = scaliing(cx,cy,x,y,r,d)
                else:
                    color = (0, 255, 0)
                    continue
            else:
                color = (0, 255, 0)
            print(x,y,r)
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

take_pic()