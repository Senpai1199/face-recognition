import numpy as np
from PIL import ImageChops, Image
import cv2

def color_confidence(frames, threshhold=60, scale=0.2):
    if len(frames) < 2:
        return False

    frame_init = frames[0][0].copy()
    startX, startY, endX, endY = frames[0][1]
    face_init = frames[0][0].copy()[startX:endX, startY:endY]
    cv2.rectangle(frame_init, (startX, startY), (endX, endY), (0,0,0), -1)

    frame_latest = frames[-1][0].copy()
    startX, startY, endX, endY = frames[-1][1]
    face_latest = frames[-1][0].copy()
    face_latest = face_latest[startX:endX, startY:endY]
    cv2.rectangle(frame_latest, (startX, startY), (endX, endY), (0,0,0), -1)

    face_diff = np.array(ImageChops.subtract(Image.fromarray(face_latest), Image.fromarray(face_init), scale))
    bg_diff = np.array(ImageChops.subtract(Image.fromarray(frame_latest), Image.fromarray(frame_init), scale))
    face_colors = np.mean(face_diff, axis=(0,1))
    bg_colors = np.mean(bg_diff, axis=(0,1))

    bg_from_black = np.linalg.norm(bg_colors)
    face_from_black = np.linalg.norm(face_colors)
    face_from_bg = np.linalg.norm(face_colors - bg_colors)
    
    #Background should be close to zero// not reflecting colors
    #Background shouldn't be reflecting more light than face
    # if bg_from_black > 45 or bg_from_black > face_from_black: 
    #     return 0

    # To check if its not just an increase in brightness
    line = np.array([1,1,1])
    dist = np.linalg.norm(np.cross(face_colors, line))/np.linalg.norm(line)
    
    if bg_from_black > 70:
        return False

    print(dist)
    if dist > 13:
        return True
    return False
    