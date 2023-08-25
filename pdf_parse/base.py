from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pdf_parse.interface import BaseInterFace


class PdfBase(BaseInterFace):
    def __init__(self, model):
        self.model = YOLO(model)

    def mkdir_pic_dir(self, filepath: str):
        path = Path(filepath)
        filename = path.stem
        Path(path.parent / filename).mkdir(exist_ok=True)
        return path.parent / filename

    def save_pic(self, img, dir_name):
        """
        保存图片
        :return:
        """

        cv2.imwrite(str(dir_name), img)

    def parse(self, filepath: str, **kwargs) -> list[Results]:
        dir_path = self.mkdir_pic_dir(filepath)

        results = self.model.predict(filepath, **kwargs)
        for result in results:
            yield self._save_results(dir_path, result)

    def _save_results(self, dir_path, result):
        pass

    def save(self, filepath: str, output_dir: str = None):
        raise NotImplemented

    def start(self):
        pass
