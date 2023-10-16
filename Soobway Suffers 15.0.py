import cv2
import mediapipe as mp
import time
from math import hypot
import pyautogui as pt
from time import time


mp_pose = mp.solutions.pose
pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=1)
mpDraw=mp.solutions.drawing_utils

camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)


'''def checkHandsJoined(image, results, draw=False, display=False):
    height, width, _ = image.shape
    
    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)

    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))
    
    if euclidean_distance < 70:
        hand_status = 'Hands Joined'
        pt.press('space')
        print("hands joined")
    else:
        hand_status = 'Hands Not Joined'''

def down(image, results,d,j):
    height, width, _ = image.shape
    
    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height)

    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * height)
    
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))
    t=time()
    if euclidean_distance < 400 and t-j>0.5:
        pt.press('down')
        d=time()
    return d

def Hchange(horizontal_position,prev):
    if prev==0 and horizontal_position==1:
        pt.press('right')
    elif prev==1 and horizontal_position==2:
        pt.press('right')
    elif prev==1 and horizontal_position==0:
        pt.press('left')
    elif prev==2 and horizontal_position==1:
        pt.press('left')
        
def checkLeftRightMid(image, results,prev):
    height, width, _ = image.shape
    horizontal_position = None
    
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)

    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)

    mid=(left_x+right_x)/2
    
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
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

    right_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    right_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * height)

    mid2=(left_x+right_x)/2

    time2=time()
    if time2> time1+0.05 :
        if (mid1-mid2)>20 and time2-d>0.35 :
            pt.press("up")
            j=time()
        mid1=mid2
        time1=time2
    return time1,mid1,j

'''print("###########")
prev=None
while camera_video.isOpened():
        
    ok, frame = camera_video.read()
        
    if not ok:
        continue
        
    frame = cv2.flip(frame, 1)    
    results=pose.process(frame)
        
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)    
        checkHandsJoined(frame, results)
        down(frame, results)
        prev = checkLeftRightMid(frame, results,prev,draw = True)
                       
    cv2.imshow('Soobway Suffers', frame)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break'''


print("############")
prev=None
time1=time4 = 0
mid1=0
d=j=0
frame_count=end_time=start_time=0
while camera_video.isOpened():
    ok, frame = camera_video.read()
        
    if not ok:
        continue

    height, width, _ = frame.shape
        
    frame = cv2.flip(frame, 1)    
    results=pose.process(frame)
        
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)    
        #checkHandsJoined(frame, results)
        d=down(frame, results,d,j)
        time1,mid1,j=jump(frame,results,time1,mid1,d,j)
        prev = checkLeftRightMid(frame, results,prev)

    cv2.line(frame, ((width*10)//27, 0), ((width*10)//27, height), (255, 255, 255), 2)
    cv2.line(frame, ((17*width)//27, 0), ((17*width)//27, height), (255, 255, 255), 2)

    time3 = time()
    if (time3 - time4) > 0:
        frames_per_second = 1.0 / (time3 - time4)
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    time4 = time3 
    cv2.imshow('Soobway Suffers', frame)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break
print("#######")    
camera_video.release()
cv2.destroyAllWindows()


