import os
import shutil

import cv2 as cv
import numpy as np

from typing import Generator, List

from torch.utils.data import Subset, DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms import v2

from custom_dataSet import CustomImageDataset


class CsvReader:
    # ref: https://rednafi.com/python/enable_repeatable_lazy_iterations/
    __file: str
    def __init__(self, file: str):
        self.__file = file

    def __iter__(self, sep=",") -> Generator:
        for row in open(file=self.__file, mode="r", encoding="UTF-8"):
            yield tuple(row.strip().split(sep))


class DataPreprocessor:
    __dataset_options: dict
    __path_options: dict
    __class_labels: List[str]  # list of label need to be duplicated & augmented
    __multipliers: List[float]  # list of corresponding multiplier for each class
    __thresholds: List[float]  # list of corresponding thresholds for each class when applying transformation

    # These fields are not somewhat necessary. Handling later
    __female_counter: int = 0  # for renaming file_name
    __male_counter: int = 0  # for renaming file_name

    def __init__(self, dataset_options: dict, path_options: dict,
                 class_labels: List[str], multipliers: List[float],
                 thresholds: List[float]):
        # Pass entire path config to this class
        self.__path_options = path_options
        self.__dataset_options = dataset_options

        self.__class_labels: List[str] = class_labels
        self.__multipliers: List[float] = multipliers
        self.__thresholds: List[float] = thresholds

        self.__female_counter: int = 0
        self.__male_counter: int = 0

    # Public methods
    def preprocessing(self):
        # Pipeline
        self.__make_annotation_file(path=self.__path_options["dataset_path"])
        # self.__resize_img()

    def split_train_test(self) -> tuple:
        dataset = CustomImageDataset(annotations_file=self.__path_options["annotation_path"],
                                     img_dir=self.__path_options["dataset_save_path"],
                                     train_size=self.__dataset_options["train_size"],
                                     )
        train_indices, test_indices = dataset.get_train_test_indices()
        train_set, test_set = Subset(dataset, train_indices), Subset(dataset, test_indices)
        return train_set, test_set

    def imgAugmentation(self, dataset):
        # Augment based on provided class_labels provided in init
        print(len(dataset))
        print(next(iter(dataset)))

        return dataset



    def getDataLoader(self, train_set, test_set) -> tuple:
        train_set = DataLoader(dataset=train_set, batch_size=self.__dataset_options["batch_size"], shuffle=True, num_workers=4,
                               pin_memory=True)
        test_set = DataLoader(dataset=test_set, batch_size=self.__dataset_options["batch_size"], shuffle=True, num_workers=4,
                              pin_memory=True)
        return train_set, test_set
    # Private methods
    def __make_annotation_file(self, path: str, gender_flag: bool = None):
        """
        Problem: Pytorch's dataloader just load directory in the format of
            file_1, file_2, file_3, ..., annotation.csv

        annotation.csv looks like:
        file_1, class_1
        file_2, class_2
        file_3, class_3
        then we add file_name,label to the first line of file
        ...

        This function is created to handle directory format in the tree shape
        E.g.
                        Root
            -----------     -----------
            |                          |
         Class 1                    Class 2
         |     |                    |      |
    CLass 1.1  Class 1.2       Class 1.3  Class 1.4
                        ...
        """
        root_path, directories, files = next(os.walk(path))

        for dir in directories:
            if root_path.split("\\")[-1] == "Male": gender_flag = False
            if root_path.split("\\")[-1] == "Female": gender_flag = True
            self.__make_annotation_file(path=os.path.join(path, dir), gender_flag=gender_flag)
        # print(path, gender_flag)

        if len(os.listdir(path)) > 0:
            if os.listdir(path)[0].endswith(".jpg"):
                for img in os.listdir(path):
                    # save info to annotation
                    # 0/ False for male - 1/ True for female
                    with open(file=self.__path_options["annotation_path"], mode="a", encoding="UTF-8") as f:

                        if gender_flag:
                            f.write(f"Female_{self.__female_counter}.jpg,{1}\n")

                            # Move & Rename img
                            shutil.copy2(src=os.path.join(path, img),
                                         dst=self.__path_options["dataset_save_path"])  # preserve metadata (timestamp, ...)

                            os.rename(src=os.path.join(self.__path_options["dataset_save_path"], img),
                                      dst=os.path.join(self.__path_options["dataset_save_path"], f"Female_{self.__female_counter}.jpg"))

                            self.__female_counter += 1

                        else:
                            f.write(f"Male_{self.__male_counter}.jpg,{0}\n")

                            # Move & Rename img
                            shutil.copy2(src=os.path.join(path, img),
                                         dst=self.__path_options["dataset_save_path"])

                            os.rename(src=os.path.join(self.__path_options["dataset_save_path"], img),
                                      dst=os.path.join(self.__path_options["dataset_save_path"], f"Male_{self.__male_counter}.jpg"))

                            self.__male_counter += 1
                    # print("Image moved:", img)


    def __resize_img(self) -> None:
        """
        Resize all image to desired size
        """
        reader = CsvReader(file=self.__path_options["annotation_path"])

        for file_name, _ in reader:
            img = cv.imread(filename=os.path.join(self.__path_options["dataset_save_path"], file_name))

            img = cv.resize(src=img,
                            dsize=(self.__dataset_options["img_size"], self.__dataset_options["img_size"]),
                            interpolation=cv.INTER_AREA
                            )

            cv.imwrite(filename=os.path.join(self.__path_options["dataset_save_path"], file_name), img=img)
        print("Resizing complete")
        return None


    def __img_class_duplication(self, class_labels: List[str], multipliers: List[float]) -> None:
        """
        :param class_labels: list of class neeed to be duplicated
               multipliers: list of corresponding duplication ratio for each class/ label
        Can handle duplicaton as many available classes as you want
        """
        csvReader = CsvReader(file=self.__path_options["annotation_path"])
        for class_label, multiplier in zip(class_labels, multipliers):
            tmp_annotation_path = os.path.join(os.getcwd(), "tmp_annotation.csv")
            total_augmented_class_img = self.__get_total_class_img(csvReader, class_label)

            # Duplicate img of specific class
            counter = 0
            while counter < multiplier * total_augmented_class_img:
                for file_name, label in csvReader:
                    if label == class_label:
                        # Duplicate img
                        img = cv.imread(filename=os.path.join(self.__path_options["dataset_save_path"], file_name))
                        filename = f'{file_name.split(".")[0]}_aug_{counter}.jpg'
                        cv.imwrite(filename=os.path.join(self.__path_options["dataset_save_path"], filename), img=img)
                        counter += 1

                        # save annotation to tmp file and then copy to annotation.csv
                        with open(file=tmp_annotation_path, mode="a", encoding="UTF-8", errors="ignore") as f:
                            f.write(f"{filename},{class_label}\n")

                    if counter >= multiplier * total_augmented_class_img:
                        break
                    print("Duplicated", counter, "images of", class_label)

            # Pass annotation from tmp.csv to annotation.csv
            with open(file=self.__path_options["annotation_path"], mode="a", encoding="UTF-8", errors="ignore") as writer:
                with open(file=tmp_annotation_path, mode="r", encoding="UTF-8", errors="ignore") as reader:
                    for row in reader:
                        writer.write(row)
            print("Write to annotation.csv complete")

            # remove tmp
            os.remove(path=tmp_annotation_path)
        del reader
        return None


    def __img_class_augmentation(self, class_labels: List[str], thresholds: List[float]) -> None:
        """
        Ref: https://manmeet3.medium.com/face-data-augmentation-techniques-ace9e8ddb030
        :param class_labels: class for augmenting
        :param threshold: level of being transformed or not
        :return:
        """
        csvreader = CsvReader(file=self.__path_options["annotation_path"])

        for class_label, threshold in zip(class_labels, thresholds):
            counter = 0  # for naming file in a specific class
            for file_name, label in csvreader:
                prob = np.random.rand(1)
                if label == class_label and prob >= threshold:
                    img = read_image(path=os.path.join(self.__path_options["dataset_save_path"], file_name))

                    if np.random.rand(1) >= .5:
                        img = v2.functional.adjust_contrast(inpt=img, contrast_factor=.5)

                    elif np.random.rand(1) < .5:
                        img = v2.functional.adjust_brightness(inpt=img, brightness_factor=.5)

                    # Normalize before saving it
                    save_image(tensor=img/255, fp=os.path.join(self.__path_options["dataset_save_path"], file_name))
                    print("Transformed", counter, "images of label", class_label)
                    counter += 1
        del csvreader
        return None

    # Static methods
    @staticmethod
    def __get_total_class_img(reader: CsvReader, class_label: str):
        return sum(1 for _, label in reader if label == class_label)










