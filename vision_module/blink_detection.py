import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def eye_aspect_ratio(eye_points, landmarks, w, h):

    points = []

    for p in eye_points:
        lm = landmarks[p]
        points.append((int(lm.x*w), int(lm.y*h)))

    p1,p2,p3,p4,p5,p6 = points

    vertical1 = np.linalg.norm(np.array(p2)-np.array(p6))
    vertical2 = np.linalg.norm(np.array(p3)-np.array(p5))
    horizontal = np.linalg.norm(np.array(p1)-np.array(p4))

    ear = (vertical1+vertical2)/(2.0*horizontal)

    return ear


closed_frames = 0
ear_history = []

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    status = "ALERT"
    color = (0,255,0)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)

            ear = (left_ear + right_ear) / 2

           
            ear_history.append(ear)
            if len(ear_history) > 5:
                ear_history.pop(0)

            ear = sum(ear_history)/len(ear_history)

            if ear < 0.23:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames < 12:
                status = "ALERT"
                color = (0,255,0)

            elif closed_frames < 30:
                status = "FATIGUE"
                color = (0,255,255)

            else:
                status = "SLEEPY"
                color = (0,0,255)

            cv2.putText(frame,
                        "Driver State: "+status,
                        (40,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        3)

            cv2.putText(frame,
                        "EAR: {:.2f}".format(ear),
                        (40,130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,255,255),
                        2)

    cv2.imshow("DriveGuard AI Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()