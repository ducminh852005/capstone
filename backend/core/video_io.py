import queue
import threading

import cv2


class ThreadedVideoReader:
    """
    Decodes frames on a background thread into a bounded queue, so H.264 decoding
    overlaps with detection work on the main thread.

    Iterating yields (frame_idx, frame) until the video ends.
    """

    _END = object()

    def __init__(self, path, start_frame=0, queue_size=8):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        if start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._start = start_frame
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        idx = self._start
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            while not self._stop.is_set():
                try:
                    self._queue.put((idx, frame), timeout=0.1)
                    break
                except queue.Full:
                    continue
            idx += 1
        self._queue.put(self._END)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is self._END:
                return
            yield item

    def release(self):
        self._stop.set()
        # Drain so the producer is never blocked on a full queue.
        while self._thread.is_alive():
            try:
                self._queue.get(timeout=0.1)
            except queue.Empty:
                pass
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
