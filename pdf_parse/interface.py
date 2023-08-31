from pathlib import Path

import cv2
from ultralytics.engine.results import Results
from ultralytics.models import YOLO


class BaseInterFace:

    def start(self):
        raise NotImplementedError

    def parse(self, result, img, **kwargs) -> list[Results]:
        raise NotImplementedError

    def save(self, img, cate_name):
        not NotImplementedError
