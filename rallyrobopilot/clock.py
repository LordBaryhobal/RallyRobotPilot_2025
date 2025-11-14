import time


class Clock:
    def __init__(self):
        self.last_t: float = time.time()

    def tick(self, fps: float):
        delta_t: float = 1 / fps
        elapsed: float = time.time() - self.last_t
        sleep: float = delta_t - elapsed
        if sleep > 0:
            time.sleep(sleep)
        self.last_t = time.time()
