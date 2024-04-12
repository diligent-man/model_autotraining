import os
import shutil
import pandas as pd

from tqdm import tqdm
from typing import Tuple
from ultralytics import YOLO
from src.data.celeb_A_dataset import CelebADataset
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2, InterpolationMode

__all__ = ["generate_celeb_A_dataset"]

SEED = 12345
NUM_WORKERS = 6
BATCH_SIZE = 384
VAL_SIZE = TEST_SIZE = 0.1

DATASET_ROOT = "/home/trong/Downloads/Dataset/Celeb_A"
CHECKPOINT_PATH = "/home/trong/Downloads/Local/Pretrained_models/YOLOv8/Face_detection/yolov8n-face.pt"


def crop_face(dataset_path: str, save_path: str, annotation_path: str,
              padding_ratio: Tuple[float] = (.1, .1, .1, .1)
              ) -> None:
    """
    padding_ratio: corresponding padding ratio for x1, x2, y1, y2
    """
    preprocessing_transformations = v2.Compose([
        v2.Resize(size=[50, 50], interpolation=InterpolationMode.NEAREST, antialias=True),
        v2.Resize(size=[224, 224], interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.GaussianBlur(kernel_size=5)
    ])

    yolo_transformations = v2.Compose([
        v2.PILToTensor(),
        v2.Resize(size=(640, 640), interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.ToDtype(torch.float32, scale=True)
    ])


    annotation = pd.read_csv(filepath_or_buffer=annotation_path)
    tmp_annotation = pd.DataFrame(columns=annotation.columns)

    yolo = YOLO(model=CHECKPOINT_PATH, task="detect").to("cuda")

    dataset = CelebADataset(dataset_path=dataset_path,
                            annotation=annotation,
                            transform=yolo_transformations
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
        predictions = yolo.predict(source=imgs, conf=.5, iou=.1, verbose=False, classes=[0], save=False)
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
                img = img[:, round(y1 * (1 - padding_ratio[2])): round(y2 * (1 + padding_ratio[3])), \
                             round(x1 * (1 - padding_ratio[0])): round(x2 * (1 + padding_ratio[1]))]

                img = preprocessing_transformations(img)

                # Save img
                filename = os.path.join(save_path, corresponding_annotation.loc[annotation_index, "image_id"])
                torchvision.io.write_jpeg(input=img, filename=filename)

                # Write annotation to tmp
                tmp_annotation = tmp_annotation._append(corresponding_annotation.loc[annotation_index, :],
                                                        ignore_index=True)
            else:
                undetectable_img += 1

        print("Undetectable img:", undetectable_img)
    tmp_annotation.to_csv(path_or_buf=f"{save_path}.csv", header=True, index=False, encoding="UTF-8")
    return None


def _add_prefix_dir(name: str):
    return f"data/{name}"

def _split_dataset(train_size: float, dataset: pd.DataFrame, split_field: str):
    return train_test_split(dataset,
                            train_size=train_size,
                            random_state=SEED,
                            shuffle=True,
                            stratify=dataset[split_field])


def make_annotation(split_field: str, annotation_path: str) -> None:
    """
    field to split into separate class
    """
    annotation: pd.DataFrame = pd.read_csv(annotation_path).loc[:, ["image_id", split_field]]
    annotation["image_id"] = annotation["image_id"].apply(_add_prefix_dir)
    class_names = annotation[split_field].unique()

    # Split train/ test
    train, test = _split_dataset(1 - TEST_SIZE, annotation, split_field)
    # Split train/ val
    train, val = _split_dataset(1 - VAL_SIZE, train, split_field)

    for dataset, name in zip((train, val, test), ("train.csv", "val.csv", "test.csv")):
        dataset.to_csv(os.path.join(DATASET_ROOT, name), index=False)
    return None


def generate_celeb_A_dataset(split_field: str) -> None:
    dataset_path = os.path.join(DATASET_ROOT, "image")
    save_path = os.path.join(DATASET_ROOT, "data")
    annotation_path = os.path.join(DATASET_ROOT, "list_attr_celeba.csv")

    if not os.path.isdir(save_path):
        os.makedirs(name=save_path, mode=0o777, exist_ok=True)
    else:
        shutil.rmtree(path=save_path)
        os.makedirs(name=save_path, mode=0o777, exist_ok=True)

    crop_face(dataset_path, save_path, annotation_path)
    make_annotation(split_field, f"{save_path}.csv")
    return None



def main() -> None:
    generate_celeb_A_dataset(split_field="Male")
    return None


if __name__ == '__main__':
    main()