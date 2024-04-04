import os
import shutil

# Counter for each class
label_counter = {
    "Female": 0,
    "Male": 0
}


def image_reordering(root: str, save_path: str, img_extension="jpg"):
    """
                       Root
            -----------     -----------
            |                          |
         Class 1                    Class 2
         |     |                    |      |
    CLass 1.1  Class 1.2       Class 1.3  Class 1.4
                        ...
    """
    root_path, directories, files = next(os.walk(root))

    for dir in directories:
        image_reordering(os.path.join(root_path, dir), save_path)

    # check whether we recurse to img destination or not
    # if true move to specified class
    for label in label_counter.keys():
        if (label in root) and os.listdir(root)[0].endswith(img_extension):
            for img in os.listdir(root):
                # Create label dir if not exists
                if not os.path.exists(os.path.join(save_path, label)):
                    os.mkdir(os.path.join(save_path, label))

                # Copy img
                shutil.copy2(src=os.path.join(root, img), dst=os.path.join(save_path, label))

                # Rename copied img
                os.rename(src=os.path.join(save_path, label, img),
                          dst=os.path.join(save_path, label, f"{label}_{label_counter[label]}.jpg"))

                # update counter
                label_counter[label] += 1

                print("Image moved:", img)


def main() -> None:
    """
    Move images from contributors to specified class directories and then make annotation.csv
    Dataset and this file must be in the same dir
    """
    save_path = os.path.join(os.getcwd(), "reordered_labelled_data")
    os.mkdir(save_path)

    root = os.path.join(os.getcwd(), "Labelled_face_attribute_dataset")
    image_reordering(root, save_path)
    return None


if __name__ == '__main__':
    main()