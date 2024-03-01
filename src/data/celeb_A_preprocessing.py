import os
import shutil
import pandas as pd

from pathlib import Path
from src.utils.multiprocessor import Multiprocessor
from sklearn.model_selection import train_test_split


def gender_separate(lower: int, upper: int,
                    src: str, dest: str,
                    field: str, dataset_type: str,
                    df: pd.DataFrame
             ) -> None:
    dest = os.path.join(dest, dataset_type)
    for class_name in ("male", "female"):
        if not os.path.isdir(os.path.join(dest, class_name)):
            os.makedirs(name=os.path.join(dest, class_name), mode=0x777, exist_ok=True)

    for i in range(lower, upper):
        if df.loc[i, field] == -1:
            # move to female
            shutil.copy2(src=os.path.join(src, df.loc[i, "image_id"]), dst=os.path.join(dest, "female"))
        else:
            # move to male
            shutil.copy2(src=os.path.join(src, df.loc[i, "image_id"]), dst=os.path.join(dest, "male"))
        print("Move", df.loc[i, "image_id"])
    return None


def class_splitting_setup(src: Path, dest: Path, df: pd.DataFrame, field: str, train_size: float = .95):
    for df, dataset_type in zip(train_test_split(df, train_size=train_size,  random_state=12345, shuffle=True), ("train", "test")):
        df.reset_index(drop=True, inplace=True)
        multiprocessor = Multiprocessor(lower=df.index[0], upper=df.index[-1] + 1,
                                        fixed_configurations=(src, dest, field, dataset_type, df),
                                        processes=os.cpu_count(),
                                        process_counter=False)
        multiprocessor(func=gender_separate)


def main() -> None:
    """
    This script is used to split dataset into train and test sets based on gender
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
    field = "Male"
    src = Path(r"D:\Dataset\Celeb_A\image")
    dest = Path(r"D:\Local\Source\python\semester_6\face_attribute\small_celeb_A")
    df = pd.read_csv(Path("D:\Dataset\Celeb_A\list_attr_celeba.csv")).loc[:5000, ["image_id", field]]
    class_splitting_setup(src, dest, df, field=field)
    return None


if __name__ == '__main__':
    main()