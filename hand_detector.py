from unittest import result
import cv2
import mediapipe as mp
import time
from google.protobuf.json_format import MessageToDict

class handDetector():
    def __init__(self,mode=False,max_hands=2,model_complex=1,detect_conf=0.5,track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complex = model_complex
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                                        self.mode,
                                        self.max_hands,
                                        self.model_complex,
                                        self.detect_conf,
                                        self.track_conf,
                                        )
        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self,input_image, draw=True):
        RGB_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(input_image)
        # print(self.results.multi_hand_landmarks)
        num_of_hands = 0
        if self.results.multi_hand_landmarks:
            num_of_hands = len(self.results.multi_hand_landmarks)
            # print(len(self.results.multi_hand_landmarks))
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(input_image,handLms,self.mp_hands.HAND_CONNECTIONS)
                    
        return input_image,num_of_hands

    def findPosition(self, input_image,hand_num=0,draw=True):
        hand_position = {}
        # Hand in image is Left or Right
        if self.results.multi_handedness:
            for idx, hand_handedness in enumerate(self.results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                hand_num = handedness_dict['classification'][0]['index']
                hand_type = handedness_dict['classification'][0]['label']
                num_hands = len(self.results.multi_hand_landmarks)
                # print('Hand No ',hand_num,'Hand Type ',hand_type)
                # print('No of Hands ',len(self.results.multi_hand_landmarks))
                if num_hands == hand_num:
                    hand_num = 0
                if self.results.multi_hand_landmarks:
                    # hand_num_len = len(self.results.multi_hand_landmarks)
                    # for hand_num in range(0,hand_num_len):
                    hand = self.results.multi_hand_landmarks[hand_num]
                    lm_list = []
                    for id, lm in enumerate(hand.landmark):
                        # print(id,lm)
                        h ,w,c = input_image.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        lm_list.append([id,cx,cy])
                        if draw:
                            cv2.circle(input_image,(cx,cy),3,(255,0,255),cv2.FILLED)
                    hand_position[hand_type] = lm_list
                    del lm_list
        return hand_position