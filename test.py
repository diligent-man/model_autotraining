# This file is used for testing syntax
import cv2 as cv
import numpy as np
import supervision as sv

from pathlib import Path
from ultralytics import YOLO

from




def main() -> None:
    dataset_path = Path(r"D:\Local\Source\python\semester_6\face_attribute\celeb_A_backup\train\male")
    save_path = Path(r"D:\Local\Source\python\semester_6\face_attribute\celeb_A\train\male")
    checkpoint_path = Path(r"D:\Local\Pretrained_models\YOLOv8\Detection\yolov8n.pt")
    yolo = YOLO(model=checkpoint_path, task="detect").to("cuda")

    for file_name in dataset_path.iterdir():
        img: np.ndarray = cv.imread(filename=file_name, flags=cv.IMREAD_UNCHANGED)

        result = sv.Detections.from_ultralytics(yolo.predict(source=img))
        print(result)
    # detection =
    #
    # coords = tuple(detection.xyxy[0].astype(int))
    # print(coords)
    # x1, y1, x2, y2 = coords
    # # h, w = y2 - y1, x2 - x1
    # img = img[y1:y2, x1:x2]
        cv.imshow("demo", img)
    #
    # # cv.imwrite(Path(r"D:\Local\Source\python\semester_6\face_attribute\result.jpg"), result)
        cv.waitKey(0)
        break

    return None


if __name__ == '__main__':
    main()