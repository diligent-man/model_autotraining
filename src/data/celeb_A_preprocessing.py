import os
import shutil
import pandas as pd
import torchvision.utils

from tqdm import tqdm
from typing import Tuple
from ultralytics import YOLO
from src.data.celeb_A_dataset import CelebADataset
from src.utils.multiprocessor import Multiprocessor
from sklearn.model_selection import train_test_split

import torch

from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import DataLoader


BATCH_SIZE = 424
NUM_WORKERS = 4
CHECKPOINT_PATH = r"D:\Local\Pretrained_models\YOLOv8\Face_detection\yolov8n-face.pt"


def crop_face(dataset_path: str, save_path: str, annotation_path: str,
              padding_ratio: Tuple[float] = (.1, .1, .1, .1)
              ) -> None:
    """
    padding_ratio: corresponding padding ratio for x1, x2, y1, y2
    """
    annotation = pd.read_csv(filepath_or_buffer=annotation_path)

    tmp_annotation_path = r"D:\Dataset\Celeb_A\tmp.csv"
    tmp_annotation = pd.DataFrame(columns=annotation.columns)

    yolo = YOLO(model=CHECKPOINT_PATH, task="detect").to("cuda")

    dataset = CelebADataset(dataset_path=dataset_path,
                            annotation=annotation,
                            transform=v2.Compose([
                                v2.Resize(size=(640, 640), interpolation=InterpolationMode.BICUBIC, antialias=True),
                                v2.PILToTensor(),
                                v2.ToDtype(torch.float32, scale=True)])
                            )

    dataloader = DataLoader(dataset=dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=True
                            )

    undetectable_img = 0
    for batch_index, imgs in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = imgs.to("cuda")
        predictions = yolo.predict(source=imgs, conf=.3, iou=.1, verbose=False, classes=[0], save=False)
        corresponding_annotation = annotation.loc[batch_index * BATCH_SIZE: min((batch_index + 1) * BATCH_SIZE, len(annotation)), :]

        for prediction, annotation_index in zip(predictions, corresponding_annotation.index):
            # Solely consider detectable image
            if len(prediction.boxes.conf) > 0:
                # Retrieve bounding box coord with highest confidence score
                # x1, y1: top left corner || x2, y2: bottom right corner
                max_conf_index = torch.argmax(prediction.boxes.conf)
                x1, y1, x2, y2 = tuple([round(value.item()) for value in prediction.boxes.xyxy[max_conf_index]])

                # HWC -> CHW (RBG img)
                img = torch.from_numpy(prediction.orig_img).permute(2, 0, 1)
                img = img[:, round(y1 * (1 - padding_ratio[2])): round(y2 * (1 + padding_ratio[3])),\
                             round(x1 * (1 - padding_ratio[0])): round(x2 * (1 + padding_ratio[1]))]

                # Save img
                torchvision.io.write_jpeg(input=img, filename=os.path.join(save_path, corresponding_annotation.loc[annotation_index, "image_id"]))

                # Write annotation to tmp
                tmp_annotation = tmp_annotation._append(corresponding_annotation.loc[annotation_index, :], ignore_index=True)
            else:
                undetectable_img += 1

        print("Undetectable img:", undetectable_img)

    tmp_annotation.to_csv(path_or_buf=tmp_annotation_path, header=True, index=False, encoding="UTF-8")
    return None


def gender_separate(lower: int, upper: int,
                    src: str, dest: str,
                    field: str, dataset_type: str,
                    df: pd.DataFrame, process_counter: int
             ) -> None:
    # Mapping function for multiprocessing
    dest = os.path.join(dest, dataset_type)
    for class_name in ("male", "female"):
        if not os.path.isdir(os.path.join(dest, class_name)):
            os.makedirs(name=os.path.join(dest, class_name), mode=0x777, exist_ok=True)

    for i in tqdm(range(lower, upper), total=upper-lower, desc=f"Moving img process {process_counter}"):
        if df.loc[i, field] == -1:
            # move to female
            shutil.copy2(src=os.path.join(src, df.loc[i, "image_id"]), dst=os.path.join(dest, "female"))
        else:
            # move to male
            shutil.copy2(src=os.path.join(src, df.loc[i, "image_id"]), dst=os.path.join(dest, "male"))
        # print("Move", df.loc[i, "image_id"])
    return None


def class_splitting_setup(dataset_path: str, save_path: str, annotation_path: str, field: str = "Male", train_size: float = .95):
    """
        This function is used to split dataset into train and test sets based on gender
        dataset ->
            train
                male
                    img_1.jpg
                    img_2.jpg
                    img_3.jpg
                    ...
                female
                   img_1.jpg
                   img_2.jpg
                   img_3.jpg
                   ...
            test
                male
                    *,jpg
                female
                    *.jpg
    """
    annotation = pd.read_csv(annotation_path).loc[:, ["image_id", field]]

    for df, dataset_type in zip(train_test_split(annotation, train_size=train_size,  random_state=12345, shuffle=True), ("train", "test")):
        df.reset_index(drop=True, inplace=True)
        multiprocessor = Multiprocessor(lower=df.index[0], upper=df.index[-1] + 1,
                                        fixed_configurations=(dataset_path, save_path, field, dataset_type, df),
                                        processes=NUM_WORKERS,
                                        process_counter=True)
        multiprocessor(func=gender_separate)


def main() -> None:
    funcs = [crop_face, class_splitting_setup]
    dataset_paths = [r"D:\Dataset\Celeb_A\image", r"D:\Dataset\Celeb_A\tmp"]
    save_paths = [r"D:\Dataset\Celeb_A\tmp", r"D:\Local\Source\python\semester_6\face_attribute\celeb_A"]
    annotation_paths = [r"D:\Dataset\Celeb_A\list_attr_celeba.csv", r"D:\Dataset\Celeb_A\tmp.csv"]

    # Check path availability
    for path in save_paths:
        if not os.path.isdir(path):
            os.makedirs(name=path, mode=0x777, exist_ok=True)
        else:
            shutil.rmtree(path=path)
            os.makedirs(name=path, mode=0x777, exist_ok=True)

    for func, dataset_path, save_path, annotation_path in zip(funcs, dataset_paths, save_paths, annotation_paths):
        func(dataset_path, save_path, annotation_path)

    # Remove tmp stuff
    shutil.rmtree(dataset_paths[1])
    os.remove(annotation_paths[1])
    return None


if __name__ == '__main__':
    main()