import random
import time
from pathlib import Path
import string

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import loguru
from ddddocr import DdddOcr

from pdf_parse.interface import BaseInterFace

project_dir = Path(__file__).parent.parent

logger = loguru.logger


class PdfBase(BaseInterFace):
    def __init__(self, model: str = None, output_dir: Path = None):

        self.ocr = DdddOcr(show_ad=False, beta=True)

        _model = project_dir / "best.pt"

        if model is not None:
            _model = model

        logger.debug("加载model:{}".format(_model))
        self.model = YOLO(_model)
        logger.debug("model加载完成:{}".format(_model))

        self.output_dir = output_dir

        # 使用自定义文件夹替代默认文件夹
        if output_dir is None:
            self.output_dir = project_dir / "images"

        # 创建文件夹
        if not self.output_dir.exists():
            self.output_dir.mkdir(exist_ok=True)

    def save(self, img, cate_name: str):
        """
        保存图片
        :return:
        """

        cate_dir = self.output_dir / cate_name

        # 创建品名文件夹
        if not cate_dir.exists():
            cate_dir.mkdir(exist_ok=True)

        # 文件名
        name = cate_dir / (str(round(time.time() * 1000)) + "".join(random.choices(string.ascii_letters, k=random.randint(4, 7))))

        suffix = "png"
        # cv2.imwrite(f"{str(name)}.{suffix}", img)
        cv2.imencode(".png", img)[1].tofile(str(name) + '.png')
        logger.info(f"[{str(name)}.{suffix}]保存成功！")

    def predict(self, img_list: list[Path], **kwargs) -> list[Results]:
        for img in img_list:
            results = self.model.predict(img, **kwargs)
            yield results

    def parse(self, result, **kwargs) -> tuple[str, bytes]:

        raise NotImplementedError

    def load_img_list(self, img_dir: str) -> list[Path]:
        dir_path = Path(img_dir)
        img_list = dir_path.glob("*.[pPjJ][nNpP][jJgG]")
        return list(img_list)

    def start(self, img_dir: str = '.'):
        img_list = self.load_img_list(img_dir)

        for results in self.predict(img_list):
            for result in results:
                if not result: continue
                for class_name, img in self.parse(result) or []:
                    self.save(img, class_name)
