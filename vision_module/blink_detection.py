import numpy as np

def eye_aspect_ratio(eye):

    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)

    return ear