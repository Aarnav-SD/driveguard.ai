import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            for lm in face_landmarks.landmark:

                h, w, c = frame.shape

                x, y = int(lm.x * w), int(lm.y * h)

                cv2.circle(frame,(x,y),1,(0,255,0),-1)

    cv2.imshow("DriveGuard Face Monitor",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()