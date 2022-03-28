import cv2
import time
# import glob
# import os
import hand_detector as hd

# width, height of the camera
w_cam, h_cam = 800, 600

# capturing the video from camera
cap = cv2.VideoCapture(0)
# Setting the width, height
cap.set(3,w_cam)
cap.set(3,h_cam)

# Reading all count images from local directory
# folder_name = 'hand_count_images'
# images_list = [cv2.imread(img_path) for img_path in glob.glob(os.path.join(folder_name,'count_*.jpg'))]

prev_time = 0
current_time = 0

# creating hand detector
detector = hd.handDetector(detect_conf=0.75)

while True:
    ret, frame = cap.read()
    # Flip it around y-axis for correct handedness output
    frame = cv2.flip(frame, 1)
    # find the hands in image using detector
    frame, num_hands = detector.findHands(frame)
    # print('Num of hands ',num_hands)
    hand_points = detector.findPosition(frame,draw=False)
    finger_coord = [(8,6),(12,10),(16,14),(20,18)]
    # thumb_coord = [(4,2)]
    up_count = 0
    up_fingers = []
    if len(hand_points) !=0:
        # print(hand_points)
        for hand_type in hand_points.keys():
            fing = []
            hand_point = hand_points[hand_type]
            if len(hand_point) != 0:
                # for thumb is open or not
                if hand_type == 'Right':
                    if num_hands ==2:
                        # if hand_point[4][1] < hand_point[2][1]:
                        if hand_point[4][1] < hand_point[3][1]:
                            # print('Right Thumb is open ')
                            up_count += 1
                            fing.append(1)
                        else:
                            # print('Right Thumb is closed ')
                            fing.append(0)
                    else:
                        # if hand_point[4][1] < hand_point[2][1]:
                        if hand_point[4][1] < hand_point[3][1]:
                            up_count += 1
                            fing.append(1)
                        else:
                            fing.append(0)
                else:
                    if num_hands == 2:
                        # if hand_point[4][1] > hand_point[2][1]:
                        if hand_point[4][1] > hand_point[3][1]:
                            # print('Left Thumb is open ')
                            up_count += 1
                            fing.append(1)
                        else:
                            # print('Left Thumb is closed ')
                            fing.append(0)
                    else:
                        # if hand_point[4][1] > hand_point[2][1]:
                        if hand_point[4][1] > hand_point[3][1]:
                            up_count += 1
                            fing.append(1)
                        else:
                            fing.append(0)
                # for fingers are open or not
                for coord in finger_coord:
                    if hand_point[coord[0]][2] < hand_point[coord[1]][2]:
                        up_count += 1
                        fing.append(1)
                    else:
                        fing.append(0)
            up_fingers.append(fing)

    # if len(up_fingers) != 0:
    #     print(up_fingers)
    if up_count != 0:
        cv2.putText(frame,f'Count :{str(up_count)}',(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(23,255,29),2)
    
    # showing finger count image
    # h_img, w_img, c_img = images_list[0].shape
    # frame[0:h_img,0:w_img] = images_list[0]

    # display the farme rate
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame,f'FPS :{int(fps)}',(570,475),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
    cv2.imshow("Image ",frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()