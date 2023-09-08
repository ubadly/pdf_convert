import random
import re
import time
from pathlib import Path
import string

import cv2
from cnocr import CnOcr
from ultralytics import YOLO
from ultralytics.engine.results import Results
import loguru
from ddddocr import DdddOcr

from pdf_parse.interface import BaseInterFace

project_dir = Path(__file__).parent.parent

logger = loguru.logger


class PdfBase(BaseInterFace):
    def __init__(self, model: str = None, output_dir: Path = None, temperature=0.9, save=False):

        # 准确度
        self.temperature = temperature

        # 保存运行
        self.is_save = save

        # self.ocr = DdddOcr(show_ad=False, beta=True)
        self.ocr = CnOcr()

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

    def save(self, img, cate_name: str, tips: str):
        """
        保存图片
        :return:
        """

        # filter cate name
        cate_name = re.sub(f"[{string.printable}]", "", cate_name)

        cate_dir = self.output_dir / cate_name

        # 创建品名文件夹
        if not cate_dir.exists():
            cate_dir.mkdir(exist_ok=True)

        # 文件名
        name = cate_dir / f'{tips}{(str(round(time.time() * 1000)) + "".join(random.choices(string.ascii_letters, k=random.randint(4, 7))))}'

        suffix = ".png"
        # cv2.imwrite(f"{str(name)}.{suffix}", img)
        cv2.imencode(suffix, img)[1].tofile(str(name) + suffix)
        logger.info(f"[{str(name)}{suffix}]保存成功！")

    def predict(self, img_list: list[Path], **kwargs) -> list[Results]:
        for img in img_list:
            results = self.model.predict(img, **kwargs)
            yield results

    def parse(self, result, **kwargs) -> tuple[str, bytes]:

        raise NotImplementedError

    def img2text(self, img):
        """
        使用ocr将图片转为text
        :param img:
        :return:
        """

        suffix = ".png"
        # 灰度化
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二值化
        ret, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imencode(suffix, binary)[1].tofile("".join(random.choices(string.ascii_letters, k=6)) + suffix)

        result = self.ocr.ocr(binary)
        return "".join(x['text'] for x in result if x['score'] > 0.7)

    def load_img_list(self, img_dir: str) -> list[Path]:
        dir_path = Path(img_dir)
        img_list = dir_path.glob("*.[pPjJ][nNpP][jJgG]")
        return list(img_list)

    def start(self, img_dir: str = '.'):
        img_list = self.load_img_list(img_dir)

        for results in self.predict(img_list,
                                    conf=self.temperature,
                                    save=self.is_save):
            for result in results:
                if not result: continue
                for class_name, img, tips in self.parse(result) or []:
                    self.save(img, class_name, tips)
