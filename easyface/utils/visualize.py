import cv2
import numpy as np


def show_image(img: np.ndarray):
    cv2.namedWindow('Real-time Face Recognition', cv2.WINDOW_NORMAL)
    cv2.imshow('Real-time Face Recognition', img[:, :, ::-1])
    cv2.resizeWindow('Real-time Face Recognition', *(640, 480))

    if cv2.waitKey(1) & 0xFF == ord('Q'):
        cv2.destroyWindow('Real-time Face Recognition')
        return False
    return True


def draw_box_and_landmark(image, box, text, landmark, box_color=(0, 0, 255)):
    # box
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), box_color, 2)
    cv2.putText(image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), lineType=cv2.LINE_AA)

    # landmarks
    cv2.circle(image, (landmark[0], landmark[1]), 1, (0, 0, 255), 4)
    cv2.circle(image, (landmark[2], landmark[3]), 1, (0, 255, 255), 4)
    cv2.circle(image, (landmark[4], landmark[5]), 1, (255, 0, 255), 4)
    cv2.circle(image, (landmark[6], landmark[7]), 1, (0, 255, 0), 4)
    cv2.circle(image, (landmark[8], landmark[9]), 1, (255, 0, 0), 4)