import cv2

from pdf_parse.base import PdfBase


class One(PdfBase):

    def parse(self, results, **kwargs):
        for result in results:
            img = result.orig_img

            for index, box in enumerate(result.boxes.xyxy):
                print("可信度", result.boxes.conf[index].tolist())
                x1, y1, x2, y2 = box.numpy().astype(int).tolist()
                # pt1 = (x1, y1)
                # pt2 = (x2, y2)

                class_name = result.names[int(result.boxes.cls[index])]
                # cv.putText(img=img, text=class_name, org=pt1, fontFace=cv.FONT_HERSHEY_SIMPLEX,
                #            fontScale=1, color=(0, 255, 0), thickness=2)
                # cv.rectangle(img=img, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=5)
                img2 = img[y1:y2, x1:x2]
                # cv.imshow("image", img2)
                # cv.imwrite("box.png", img)
                yield class_name, img2


if __name__ == '__main__':
    One().start(r"C:\Users\Administrator\Downloads\zhongyaotupu")
