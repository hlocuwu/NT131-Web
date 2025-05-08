# Environment Import
import cv2
import mediapipe as mp
import numpy as np
import requests

# declear and import data
mp_drawing =mp.solutions.drawing_utils
mp_drawing_styles =mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Parameter Define & initialize
#record all the 33 different key landmarks in each frame of the video
dir1={}
#Key for the video parameter
flag=0
#Count how many fall frame happened
counter = 0
#Initialize the body angle to the camera
body_angle='front'

#body position parameter
sideway_slight=0
sideway_whole=0
front = 0
# Fall parameters
fall=0


# Video Import for testing
# the path of video to import
video_path = r'C:\Users\22095\Desktop\Third year project\UR Fall Detection Dataset\Normal.mp4'
cap = cv2.VideoCapture(video_path)


# Real-time video import

#Depend on which camera device we are using: 0 is main camera; 1 is the external camera 
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)


# Fallen and Falling detection
with mp_pose.Pose(min_detection_confidence=0.6,min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
    #while True:
        success,image = cap.read()
        if image is None:
            break
        if not success:
            print('Ignoring empty camera frame')
            #loading a video, use break
            #real-time, use continue
            break
           #continue
        #To improve the performance, change the image to note writeable
        image.flags.writeable = False
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        #Make a judgement to see whether recongized the landmark
        lst=[]
        p_lst = results.pose_landmarks
        #check whether there is an input
        if p_lst is None:
            print('no point is founded')
            continue
        else:
            for i in results.pose_landmarks.landmark:
                lst.append((i.x, i.y, i.z, i.visibility))
        h, w, _ = image.shape


 # Parameter settings for fallen detection
        shoulder_wide = abs(lst[11][0] - lst[12][0])
        print(lst[23][1],lst[24][1],'\t',lst[11][1],lst[12][1])
        
        #the height of the shoulder to hip
        s_h_high = abs((lst[23][1]+lst[24][1]-lst[11][1]-lst[12][1])/2)

        #the length between the shoudler and the hip
        s_h_long=np.sqrt(((lst[23][1]+lst[24][1]-lst[11][1]-lst[12][1] )/2)**2+((lst[23][0]+lst[24][0]-lst[11][0]-lst[12][0] )/2)**2)

        #the height of the hip to feet
        h_f_high = ((lst[28][1]+lst[27][1]-lst[24][1]-lst[23][1])/2)
         
        #the length between the hip and the feet
        #这是在两只脚与肩膀同宽的情况下，如果可以需要把这个设为定值
        h_f_long=np.sqrt(((lst[28][1]+lst[27][1]-lst[24][1]-lst[23][1] )/2)**2+((lst[28][0]+lst[27][0]-lst[24][0]-lst[23][0] )/2)**2)
        
    # Parameters for avoiding bow detect as fall
    #=========================================================================
        #the length between hip and the palm which is the ground
        h_g_high = abs((lst[23][1] + lst[24][1] - lst[29][1] - lst[30][1])/2)
        
        h_g_long=np.sqrt(((lst[32][1]+lst[31][1]-lst[24][1]-lst[23][1] )/2)**2+((lst[32][0]+lst[27][0]-lst[31][0]-lst[23][0] )/2)**2)
        
        #the length between shoulder and the ground
        s_g_high = abs((lst[11][1] + lst[12][1] - lst[29][1] - lst[30][1])/2)
    #=========================================================================    
        rate1=shoulder_wide/s_h_high

# determine the orientation to the camera
# the orientation to the camera is crucial that becouse usually we need to determine the orientation has a huge influence to the fallen posture
        if 0.2< rate1 < 0.4:
            sideway_slight +=1
            sideway_whole = 0
            front = 0
        elif rate1< 0.2 :
            sideway_whole+=1
            sideway_slight=0
            front = 0
        else:
            sideway_whole  = 0
            sideway_slight = 0
            front = 0
        if sideway_slight >= 3:
            print(f'sideway slight')
            sideway_slight = 0
            body_angle = 'sideway slight'
        elif sideway_whole >= 3:
            print(f'sideway whole')
            sideway_whole = 0
            body_angle = 'sideway whole'
        else:
            front += 1
        if front >= 3:
            body_angle = 'front'
            front = 0
            print('front')
            
                        
        print('s_h_high: ',s_h_high, 's_h_long: ',s_h_long,'h_f_high: ', h_f_high,'h_f_long: ')

#Fall detection algorithm parameters
#==================================================================================
        para_s_h_1 = 1.15
        para_s_h_2 = 0.85
        para_h_f = 0.6
        para_fall_time = 5

#Fall detection Step 1
#first Part test code for detect Not Fall
#==================================================================================
        if s_h_high < s_h_long*para_s_h_1 and s_h_high > s_h_long*para_s_h_2:
            print(f'Not Fall')
            fall = 0

#Fall detection Step 2
#==================================================================================
        elif h_f_high < para_h_f * h_f_long:
            fall+=1
        else:
            fall=0
            print(f'Bend Over')
        if fall>=para_fall_time:
            print(f'fall')
            counter += 1
            
           #Call Push Deer's API to use Push Deer's notification push function
           #api="https://api2.pushdeer.com/message/push?pushkey=<the key of mobile device> & text= Fall  "
           #req = requests.post(api)

            fall=0
            print(lst[0][1],'\t',lst[11][1],'\t',lst[23][1])
        print('============================================================================================================')           
            
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )    

#Falling detection
#Different frame comparation
#================================================================================== 
        #contain frames with 33 different keypoints information
        dir1[str(flag)]=(lst)
        #time parameter
        flag+=1
        

        if len(dir1) % 30 == 0:
            #choose frames in a certain constant interval to include more information
            interval = 1
            time_num = len(dir1)//interval
            #Calculate landmarks in each second
            start_num = len(dir1) - 30
            #initialize the falling counter
            falling1 = 0
            #test
            print(time_num)

            for i in range(start_num+6 ,time_num):

#Falling detection algorithm parameters
#==================================================================================
                now_lst = dir1[str(i*interval)]
                pre_lst = dir1[str((i-6)*interval)]

                s_h_high = (pre_lst[23][1] - pre_lst[11][1] + pre_lst[24][1] - pre_lst[12][1])/2
                s_h_long=np.sqrt(((pre_lst[23][1]+pre_lst[24][1]-pre_lst[11][1]-lst[12][1] )/2)**2+((pre_lst[23][0]+pre_lst[24][0]-pre_lst[11][0]-pre_lst[12][0] )/2)**2)

                para_falling_s_h_1 = 1.15
                para_falling_s_h_2 = 0.85
                para_v_1 = 0.5
                para_v_2 = 0.3
                para_falling_time = 3
                


#Falling detection
#First Part test code for detect Not Falling
#==================================================================================
                if s_h_high < s_h_long*para_falling_s_h_1 and s_h_high > s_h_long*para_falling_s_h_2:
                    print(f'not falling')
                elif now_lst[0][1] < para_v_1*((pre_lst[11][1] + pre_lst[12][1])/2):
                    print(f'falling step 1')
                    falling1 +=1

                #     if now_lst[0][1] < para_v_2*((pre_lst2[11][1] + pre_lst2[12][1])/2):
                #         print(f'falling step 2')
                #         falling2 +=1

                    if falling1 >= para_falling_time:

                        print(f'falling situation now')
                        #refresh falling counter
                        falling1= 0
                

                        #Call Push Deer's API to use Push Deer's notification push function
                        #api="https://api2.pushdeer.com/message/push?pushkey=<the key of mobile device> & text= Falling  "
                        #req = requests.post(api)
                        
                else:
                    print(f'not falling in this second')
                    #refresh falling counter
                    falling1=0
            #release the space of dictionary
            dic1 = {}

            
        #setup status box
        cv2.rectangle(image,(0,0),(225,130),(245,117,16),-1)
        
        # rep data
        cv2.putText(image,'Fall Frames Number',(15,12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,'Body Angle',(15,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(10,65),
                     cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,str(body_angle),(10,110),
                     cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)

                    
                
        cv2.imshow('Mediapipe Feed', image)
            
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
