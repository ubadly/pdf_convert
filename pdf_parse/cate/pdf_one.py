import cv2

from pdf_parse.interface import PdfBase


class One(PdfBase):
    def __init__(self):
        super().__init__(r"D:\work\pdf_convert\best.pt")

    def save(self, filepath: str, output_dir: str = None):
        results = self.parse(filepath)
        for dir_path, result in results:
            img = result.orig_img

            for index, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = box.numpy().astype(int).tolist()
                pt1 = (x1, y1)
                pt2 = (x2, y2)

                class_name = result.names[int(result.boxes.cls[index])]
                cv2.putText(img=img, text=class_name, org=pt1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 255, 0), thickness=2)
                cv2.rectangle(img=img, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=5)
                img2 = img[y1:y2, x1:x2]
                # cv.imshow("image", img2)
                self.save_pic(img2, f"{dir_path}/{index}.png")


if __name__ == '__main__':
    One()
