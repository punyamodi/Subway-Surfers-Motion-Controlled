import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from math import hypot
import cv2
import time
import time
import pyautogui as pt
import webbrowser
import keyboard
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def checkHandsJoined(image, keypoints_with_scores):
    height, width, _ = image.shape
    
    left_wrist_landmark = keypoints_with_scores[0][0][9][0:2]*[height,width]

    right_wrist_landmark = keypoints_with_scores[0][0][10][0:2]*[height,width]
    
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))
    
    if euclidean_distance < 100:
        pt.press('space')


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

def display_fps(frame, time1):
    start_time = time.time()
    elapsed_time = start_time - time1
    time1 = start_time
    fps = 1.0 / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return time1

def Hchange(horizontal_position,prev):
    if prev==0 and horizontal_position==1:
        pt.press('right')
    elif prev==1 and horizontal_position==2:
        pt.press('right')
    elif prev==1 and horizontal_position==0:
        pt.press('left')
    elif prev==2 and horizontal_position==1:
        pt.press('left')

def checkLeftRightMid(image,keypoints_with_scores,prev):
    height, width, _ = image.shape
    mid=int(keypoints_with_scores[0][0][0][1:2]*[width])
    
    if (mid<=(width*10)//27):
        horizontal_position = 0
        Hchange(horizontal_position,prev)
        
    elif (mid>=(17*width)//27):
        horizontal_position = 2
        Hchange(horizontal_position,prev)
        
    elif (mid>=(width*10)//27 and mid<=(17*width)//27):
        horizontal_position = 1
        Hchange(horizontal_position,prev)
    return horizontal_position
        
def jump(image,results,time1,mid1,d,j):
    height, width, _ = image.shape
    mid2=int(keypoints_with_scores[0][0][0][0:1]*[height])
    time2=time.time()
    if time2> time1+0.05 :
        if (mid1-mid2)>12 and time2-d>0.35 :
            pt.press("up")
            j=time.time()
        mid1=mid2
        time1=time2
    return time1,mid1,j       

def down(image,results,d,j):
    height, width, _ = image.shape
    nose = keypoints_with_scores[0][0][0][0:2]*[height,width]
    right_wrist_landmark = keypoints_with_scores[0][0][11][0:2]*[height,width]
    
    euclidean_distance = int(hypot(nose[0] - right_wrist_landmark[0],nose[1] - right_wrist_landmark[1]))
    t=time.time()
    if euclidean_distance < 150 and t-j>0.5:
        pt.press('down')
        d=time.time()
    return d

        
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 960)

time1=timea=mid1=start_time=0
prev=1
d=j=0
special_key = 'n'
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.float32)
    
     
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
     
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    time1=display_fps(frame,time1)
    d=down(frame,keypoints_with_scores,d,j)
    timea,mid1,j=jump(frame,keypoints_with_scores,timea,mid1,d,j)
    height, width, _ = frame.shape
    checkHandsJoined(frame,keypoints_with_scores)
    prev = checkLeftRightMid(frame,keypoints_with_scores,prev)
    cv2.line(frame, ((width*10)//27, 0), ((width*10)//27, height), (0, 0, 255), 4)
    cv2.line(frame, ((17*width)//27, 0), ((17*width)//27, height), (0, 0, 255), 4)

    cv2.imshow('MoveNet Lightning', frame)

                
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

