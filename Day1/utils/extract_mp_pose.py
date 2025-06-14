import cv2
import mediapipe as mp
from tqdm import tqdm
import numpy as np
import os


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

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


def extract_keypoints(vidf, save_video=True, model_complexity=2):    
    """
    Extracts keypoints from a video file using MediaPipe Holistic.
    Args:
        vidf (str): Path to the video file.
    """
    mp_holistic = mp.solutions.holistic
    # check if the vidf is 0 (webcam input)
    if vidf == 0:
        print("Using webcam as input")
        capture = cv2.VideoCapture(vidf)
        # in that case, we use the following path for saving the output
        video_name = "webcam"
        # get project root directory
        video_path = os.path.join(os.getcwd(), 'test_videos')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        vidf = os.path.join(video_path, video_name + '.mp4')
        output_path = os.path.join(video_path, video_name + '.npy')
        video_output_path = os.path.join(video_path, video_name + '_output.mp4')
    else:
        capture = cv2.VideoCapture(vidf)
    video_name = vidf.split('/')[-1].split('.')[0]
    video_path = os.path.dirname(vidf)
    # save the keypoints in the same directory as the video file
    output_path = os.path.join(video_path, video_name + '.npy')
    video_output_path = os.path.join(video_path, video_name + '_output.mp4')
    # get the number of frames in the video
    frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    samplerate = capture.get(cv2.CAP_PROP_FPS)
    print(f"Video resolution: {frameWidth}x{frameHeight}, FPS: {samplerate}")
    # get the number of frames in the video
    if video_name == "webcam":
        print("Using webcam input, setting number of frames to 1000")
        num_frames = 1000
    else:
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the number of frames in the video
    print(f"Number of frames in the video: {num_frames}")
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(video_output_path, fourcc, 
                            fps=samplerate, frameSize=(int(frameWidth), int(frameHeight)))
        
    with mp_holistic.Holistic(static_image_mode=False,           # Video stream mode 
    model_complexity=model_complexity,                # Highest-accuracy pose model 
    refine_face_landmarks=False,        # Finer facial detail (iris, contours) 
    enable_segmentation=False,          # Person mask for effects
    smooth_landmarks=True,             # Temporal smoothing to reduce jitter
    min_detection_confidence=0.7,      # Filter weak detections
    min_tracking_confidence=0.7        # Filter unstable tracks
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
                    arm_connection_thickness=2,
                    draw=True
                )
            if save_video:
                out.write(image)
            all_kpts.append(kpts)

            cv2.imshow("merged_landmarks", image)

            if cv2.waitKey(1) == 27:
               break
            cv2.waitKey(1)
        capture.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        cv2.destroyWindow("merged_landmarks")
        cv2.waitKey(1)
        
    all_kpts = np.array(all_kpts)
    # Save the keypoints as npy array
    np.save(output_path, all_kpts)
    return_dict = {
        'keypoints': all_kpts,
        'samplerate': samplerate,
        'video_name': video_name,
        'video_path': vidf,
         'output_path': output_path,
         'video_output_path': video_output_path
    }
    return return_dict
if __name__ == "__main__":
    vidf = "test_videos/tedtalk.webm"
    extract_keypoints(vidf=vidf, save_video=True)
    # Save the keypoints as npy array
    # np.save(output_path, all_kpts)
    # return_dict = {
    #     'keypoints': all_kpts,
    #     'samplerate': samplerate,
    #     'video_name': video_name,
    #     'video_path': video_path
    # }
    # return return_dict
    # return_dict = extract_keypoints(vidf=vidf, save_video=True)
    # print(return_dict)