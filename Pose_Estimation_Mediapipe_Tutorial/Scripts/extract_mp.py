
import mediapipe as mp #mediapipe
import cv2 #opencv
import math #basic operations
import numpy as np #basic operations
import pandas as pd #data wrangling
import csv #csv saving
import os #some basic functions for inspecting folder structure etc.

import glob
mypath = "" #this is your folder with (all) your video(s)
#time series output folder
inputfol = ""
outputf_mask = "./Output_Videos/"
outtputf_ts = "./Output_TimeSeries/"
# create output folders if they do not exist
if not os.path.exists(outputf_mask):
    os.makedirs(outputf_mask)
if not os.path.exists(outtputf_ts):
    os.makedirs(outtputf_ts)

#check videos to be processed
print("The following folder is set as the output folder where all the pose time series are stored")
print(os.path.abspath(outtputf_ts))
print("\n The following folder is set as the output folder for saving the masked videos ")
print(os.path.abspath(outputf_mask))
print("\n The following video(s) will be processed for masking: ")


#load in mediapipe modules
mp_holistic = mp.solutions.holistic
# Import drawing_utils and drawing_styles.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
 
##################FUNCTIONS AND OTHER VARIABLES
#landmarks 33x that are used by Mediapipe (Blazepose)
markersbody = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
          'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
          'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
          'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
          'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
costume_markers = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
          'RIGHT_ELBOW']
markershands = ['LEFT_WRIST', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP', 'LEFT_INDEX_FINGER_MCP',
              'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP', 'LEFT_MIDDLE_FINGER_MCP', 
               'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP', 'LEFT_RING_FINGER_MCP', 
               'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP', 'LEFT_PINKY_FINGER_MCP', 
               'LEFT_PINKY_FINGER_PIP', 'LEFT_PINKY_FINGER_DIP', 'LEFT_PINKY_FINGER_TIP',
              'RIGHT_WRIST', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP', 'RIGHT_INDEX_FINGER_MCP',
              'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP', 'RIGHT_MIDDLE_FINGER_MCP', 
               'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP', 'RIGHT_RING_FINGER_MCP', 
               'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP', 'RIGHT_PINKY_FINGER_MCP', 
               'RIGHT_PINKY_FINGER_PIP', 'RIGHT_PINKY_FINGER_DIP', 'RIGHT_PINKY_FINGER_TIP']
facemarks = [str(x) for x in range(478)] #there are 478 points for the face mesh (see google holistic face mesh info for landmarks)

print("Note that we have the following number of pose keypoints for markers body")
print(len(markersbody))

print("\n Note that we have the following number of pose keypoints for markers hands")
print(len(markershands))

print("\n Note that we have the following number of pose keypoints for markers face")
print(len(facemarks ))

#set up the column names and objects for the time series data (add time as the first variable)
markerxyzbody = ['time']
markerxyzhands = ['time']
markerxyzface = ['time']

for mark in markersbody:
    for pos in ['X', 'Y', 'Z', 'visibility']: #for markers of the body you also have a visibility reliability score
        nm = pos + "_" + mark
        markerxyzbody.append(nm)
for mark in markershands:
    for pos in ['X', 'Y', 'Z']:
        nm = pos + "_" + mark
        markerxyzhands.append(nm)
for mark in facemarks:
    for pos in ['X', 'Y', 'Z']:
        nm = pos + "_" + mark
        markerxyzface.append(nm)

#check if there are numbers in a string
def num_there(s):
    return any(i.isdigit() for i in s)

#take some google classification object and convert it into a string
def makegoginto_str(gogobj):
    gogobj = str(gogobj).strip("[]")
    gogobj = gogobj.split("\n")
    return(gogobj[:-1]) #ignore last element as this has nothing

#make the stringifyd position traces into clean numerical values
def listpostions(newsamplemarks):
    newsamplemarks = makegoginto_str(newsamplemarks)
    tracking_p = []
    for value in newsamplemarks:
        if num_there(value):
            stripped = value.split(':', 1)[1]
            stripped = stripped.strip() #remove spaces in the string if present
            tracking_p.append(stripped) #add to this list  
    return(tracking_p)



mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_and_save_custom_landmarks(image, results, skip_pose_ids=None,
                          hand_landmark_style=None,
                          hand_connection_style=None,
                          pose_point_radius=4,
                          pose_point_color=(0,255,0),
                          pose_connection_style=None,
                          connect_hands_to_body=True,
                          arm_connection_color=(255,0,0),
                          arm_connection_thickness=2, 
                          draw=False):
    h, w, _ = image.shape
    """
    Draws all hand landmarks + filtered pose landmarks on `image`.

    Args:
      image:       BGR image to draw onto.
      results:     Holistic.process(...) results.
      skip_pose_ids: set of mp_holistic.PoseLandmark to omit.
      hand_landmark_style, hand_connection_style:
        DrawingSpec for hands (defaults to MP styles).
      pose_point_radius, pose_point_color:
        circle style for filtered pose points.
      pose_connection_style:
        DrawingSpec for pose connections (defaults to green, thickness=2).
    """
    skip_pose_ids = skip_pose_ids or set()
    # default styles
    hand_landmark_style    = hand_landmark_style    or mp_styles.get_default_hand_landmarks_style()
    hand_connection_style  = hand_connection_style  or mp_styles.get_default_hand_connections_style()
    pose_connection_style  = pose_connection_style  or mp_drawing.DrawingSpec(color=pose_point_color, thickness=2)

    # 1) draw **all** hand landmarks
    frame_keypoints = []
    if results.left_hand_landmarks:
        if draw:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_style,
                connection_drawing_spec=hand_connection_style
            )
        # add left hand keypoints to frame_keypoints
        for idx, lm in enumerate(results.left_hand_landmarks.landmark):
            frame_keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
    else:
        # If no left hand landmarks, add placeholders
        for i in range(21):
            frame_keypoints.append([0, 0, 0, 0])
    
    if results.right_hand_landmarks:
        if draw:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_style,
                connection_drawing_spec=hand_connection_style
            )
        # add right hand keypoints to frame_keypoints
        for idx, lm in enumerate(results.right_hand_landmarks.landmark):
            frame_keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
    else:
        # If no right hand landmarks, add placeholders
        for i in range(21):
            frame_keypoints.append([0, 0, 0, 0])
    # 2) draw **filtered** pose points & connections
    if results.pose_landmarks:
        h, w, _ = image.shape
        if draw:
            # draw the points (skip any in skip_pose_ids)
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                landmark = mp_holistic.PoseLandmark(idx)
                if landmark in skip_pose_ids:
                    continue
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), pose_point_radius, pose_point_color, -1)
            

        
        
        # add filtered connections to frame_keypoints
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in skip_pose_ids:
                continue
            else:
                frame_keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
            

        # draw connections
        if draw:
            # build filtered connections
            filtered_conns = [
                (start, end)
                for (start, end) in mp_holistic.POSE_CONNECTIONS
                if (mp_holistic.PoseLandmark(start) not in skip_pose_ids and
                    mp_holistic.PoseLandmark(end  ) not in skip_pose_ids)
            ]
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                filtered_conns,
                landmark_drawing_spec=None,            # already drew circles
                connection_drawing_spec=pose_connection_style
            )
            # 3) optionally connect each hand’s wrist back to its elbow
            if connect_hands_to_body and results.pose_landmarks:
                # LEFT
                if results.left_hand_landmarks:
                    l_wrist = results.left_hand_landmarks.landmark[0]
                    l_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
                    p1 = (int(l_wrist.x * w), int(l_wrist.y * h))
                    p2 = (int(l_elbow.x * w), int(l_elbow.y * h))
                    cv2.line(image, p1, p2, arm_connection_color, arm_connection_thickness)

                # RIGHT
                if results.right_hand_landmarks:
                    r_wrist = results.right_hand_landmarks.landmark[0]
                    r_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                    p1 = (int(r_wrist.x * w), int(r_wrist.y * h))
                    p2 = (int(r_elbow.x * w), int(r_elbow.y * h))
                    cv2.line(image, p1, p2, arm_connection_color, arm_connection_thickness)
    else:   
        # If no pose landmarks, add placeholders
        for i in range(len(costume_markers)):
            frame_keypoints.append([0, 0, 0, 0])

    return image, frame_keypoints
 

import mediapipe as mp
from tqdm import tqdm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Which PoseLandmark indices we want to skip because
SKIP_POSE_IDS = {
    mp_holistic.PoseLandmark.LEFT_WRIST,
    mp_holistic.PoseLandmark.RIGHT_WRIST,
    mp_holistic.PoseLandmark.LEFT_PINKY,
    mp_holistic.PoseLandmark.RIGHT_PINKY,
    mp_holistic.PoseLandmark.LEFT_INDEX,
    mp_holistic.PoseLandmark.RIGHT_INDEX,
    mp_holistic.PoseLandmark.LEFT_THUMB,
    mp_holistic.PoseLandmark.RIGHT_THUMB,
    mp_holistic.PoseLandmark.LEFT_HIP,
    mp_holistic.PoseLandmark.RIGHT_HIP,
    mp_holistic.PoseLandmark.LEFT_KNEE,
    mp_holistic.PoseLandmark.RIGHT_KNEE,
    mp_holistic.PoseLandmark.LEFT_ANKLE,
    mp_holistic.PoseLandmark.RIGHT_ANKLE,
    mp_holistic.PoseLandmark.LEFT_HEEL,
    mp_holistic.PoseLandmark.RIGHT_HEEL,
    mp_holistic.PoseLandmark.LEFT_FOOT_INDEX,
    mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX,
    mp_holistic.PoseLandmark.NOSE,
    mp_holistic.PoseLandmark.LEFT_EYE_INNER,
    mp_holistic.PoseLandmark.LEFT_EYE,
    mp_holistic.PoseLandmark.LEFT_EYE_OUTER,
    mp_holistic.PoseLandmark.RIGHT_EYE_OUTER,
    mp_holistic.PoseLandmark.RIGHT_EYE,
    mp_holistic.PoseLandmark.RIGHT_EYE_INNER,
    mp_holistic.PoseLandmark.RIGHT_EYE_OUTER,
    mp_holistic.PoseLandmark.LEFT_EAR,
    mp_holistic.PoseLandmark.RIGHT_EAR,
    mp_holistic.PoseLandmark.MOUTH_LEFT,
    mp_holistic.PoseLandmark.MOUTH_RIGHT, 
}
# Process videos
for vidf in vfiles:
    print(f"Processing video: {vidf}")
    print(f"Video {vfiles.index(vidf)+1} of {len(vfiles)}")
    
    videoname = vidf
    videoloc = videoname
    capture = cv2.VideoCapture(videoloc)
    # get the number of frames in the video
    frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    samplerate = capture.get(cv2.CAP_PROP_FPS)
    # get the number of frames in the video
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the number of frames in the video
    print(f"Number of frames in the video: {num_frames}")
    with mp_holistic.Holistic(static_image_mode=False,           # Video stream mode :contentReference[oaicite:7]{index=7}
    model_complexity=2,                # Highest-accuracy pose model :contentReference[oaicite:8]{index=8}
    refine_face_landmarks=False,        # Finer facial detail (iris, contours) :contentReference[oaicite:9]{index=9}
    enable_segmentation=False,          # Person mask for effects :contentReference[oaicite:10]{index=10}
    smooth_landmarks=True,             # Temporal smoothing to reduce jitter :contentReference[oaicite:11]{index=11}
    min_detection_confidence=0.1,      # Filter weak detections :contentReference[oaicite:12]{index=12}
    min_tracking_confidence=0.1        # Filter unstable tracks :contentReference[oaicite:13]{index=13}
    ) as holistic:
        all_kpts = []
        for i in tqdm(range(num_frames), desc="Processing frames", unit="frame"):
            
            ret, frame = capture.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            h, w, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image, kpts = draw_and_save_custom_landmarks(
                    image,
                    results,
                    skip_pose_ids=SKIP_POSE_IDS,
                    pose_point_radius=5,
                    pose_point_color=(0,255,0),
                    connect_hands_to_body=True,
                    arm_connection_color=(255,0,0),       # red lines for the “arm” link
                    arm_connection_thickness=2
                )
            all_kpts.append(kpts)
            if True:
                if cv2.waitKey(1) == 27:
                    break
                cv2.imshow("merged_landmarks", image)
                cv2.waitKey(1)
                capture.release()
                cv2.destroyAllWindows()
    all_kpts = np.array(all_kpts)
    # Save the keypoints as npy array
    video_name = vidf.split('/')[-1].split('.')[0]
    np.save(outtputf_ts + video_name+ '.npy', all_kpts)
    # 
