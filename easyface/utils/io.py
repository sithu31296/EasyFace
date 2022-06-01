import cv2
import time
from threading import Thread


class WebcamStream:
    def __init__(self, device_id: int) -> None:
        self.frame = None
        self.device_id = device_id
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(3, 640)
        self.capture.set(4, 480)
        self.thread = Thread(target=self._read_frames)

    def _read_frames(self):
        while True:
            ret, frame = self.capture.read()
            if not ret: break
            self.frame = frame[:, :, ::-1]

    def __iter__(self):
        self.thread.start()
        time.sleep(1)
        return self

    def __next__(self):
        if not self.capture.isOpened():
            raise StopIteration
        return self.frame.copy()

    def stop(self):
        self.capture.release()