import cv2
import time
import torch
import numpy as np
from threading import Thread
from torchvision.io import read_video, write_video


class WebcamStream:
    def __init__(self, device_id: int) -> None:
        self.frame = None
        self.device_id = device_id
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        assert self.capture.isOpened(), f"Failed to open webcam {device_id}"
        self.capture.set(3, 640)
        self.capture.set(4, 480)
        self.thread = Thread(target=self._read_frames)

    def _read_frames(self):
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret: break
            self.frame = frame[:, :, ::-1]

    def __iter__(self):
        self.thread.start()
        time.sleep(1)
        return self

    def __next__(self):
        if cv2.waitKey(1) == ord("q"):
            self.stop()
        return self.frame.copy()

    def stop(self):
        cv2.destroyAllWindows()
        self.capture.release()
        raise StopIteration


class VideoReader:
    def __init__(self, video_file: str) -> None:
        self.frames, _, info = read_video(video_file, pts_unit='sec')
        self.fps = info['video_fps']

        print(f"Processing '{video_file}'...")
        print(f"Total Frames: {len(self.frames)}")
        print(f"Video Size  : {list(self.frames.shape[1:-1])}")
        print(f"Video FPS   : {self.fps}")

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.frames)

    def __next__(self):
        if self.count == len(self.frames):
            raise StopIteration
        frame = self.frames[self.count]
        self.count += 1
        return frame.numpy()


class VideoWriter:
    def __init__(self, file_name, fps):
        self.fname = file_name
        self.fps = fps
        self.frames = []

    def update(self, frame):
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame.copy())
        self.frames.append(frame)

    def write(self):
        print(f"Saving video to '{self.fname}'...")
        write_video(self.fname, torch.stack(self.frames), self.fps)


class FPS:
    def __init__(self, avg=10) -> None:
        self.accum_time = 0
        self.counts = 0
        self.avg = avg

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self):
        self.synchronize()
        self.prev_time = time.time()

    def stop(self, debug=True):
        self.synchronize()
        self.accum_time += time.time() - self.prev_time
        self.counts += 1
        if self.counts == self.avg:
            self.fps = round(self.counts / self.accum_time)
            if debug: print(f"FPS: {self.fps}")
            self.counts = 0
            self.accum_time = 0