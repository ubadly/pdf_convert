import cv2
import numpy as np
from PIL.Image import Image

from pdf_parse.base import PdfBase


class Row(PdfBase):

    def parse(self, result, **kwargs):
        img = result.orig_img

        # 存放图片
        titles = list()
        images = list()

        for index, box in enumerate(result.boxes.xyxy):
            # 获取图片坐标
            x1, y1, x2, y2 = box.numpy().astype(int).tolist()

            # 获取款类型
            class_str = result.names[int(result.boxes.cls[index])]

            # 根据坐标分割图片
            img2 = img[y1:y2, x1:x2]

            # 存放标题和图片信息
            if class_str == 'title':

                # 将BGR类型的图片转换为RGB的类型

                # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                #
                # img2 = cv2.imencode(".png", img2)[1]
                # 识别title
                class_name = self.img2text(img2)
                titles.append({
                    "title": class_name,
                    "pos": (x1, y1, x2, y2)

                })
            elif class_str == 'imgBord':
                images.append({
                    "img": img2,
                    "pos": (x1, y1, x2, y2)
                })

        # 根据定位对应上title和图片
        if len(titles) != len(images): return

        # 标注
        tips = ""
        # 识别图片在同一行
        for title in titles:
            for image in images:

                if image['pos'][3] >= title['pos'][1] >= image['pos'][1] or image['pos'][3] >= title['pos'][3] >= image['pos'][1]:
                    # print(title, image)
                    # cv2.imshow("title['title']", image['img'])
                    # print("================================================================")
                    # cv2.waitKey(0)
                    yield title['title'], image['img'], tips


if __name__ == '__main__':
    Row().start(r"C:\Users\Administrator\Downloads\zhongyaotupu")
