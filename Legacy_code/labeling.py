import os
import shutil
import cv2 as cv

raw_data_path = os.path.join(os.getcwd(), "raw_data")

# gender
male_path = os.path.join(os.getcwd(), "Male")
female_path = os.path.join(os.getcwd(), "Female")

# emotion
male_positive = os.path.join(male_path, "Positive")
male_neutral = os.path.join(male_path, "Neutral")
male_negative = os.path.join(male_path, "Negative")

female_positive = os.path.join(female_path, "Positive")
female_neutral = os.path.join(female_path, "Neutral")
female_negative = os.path.join(female_path, "Negative")


def gender_labeller() -> None:
    for filename in sorted(os.listdir(raw_data_path)):
        img = cv.imread(os.path.join(raw_data_path, filename), cv.IMREAD_COLOR)
        # img = cv.resize(img, dsize=(250, 250), interpolation=cv.INTER_LANCZOS4)
        cv.imshow("Male/ Female", img)

        k = cv.waitKey(0)
        if k == ord('q'):
            # stop programme
            cv.destroyAllWindows()
            break
        elif k == ord('b'):
        # elif k == _Getch("\x1b[C"):
            shutil.move(os.path.join(raw_data_path, filename), male_path)
            print("Move to Male")
        elif k == ord('m'):
            shutil.move(os.path.join(raw_data_path, filename), female_path)
            print("Move to Female")
        elif k == ord('n'):
            os.remove(os.path.join(raw_data_path, filename))
            print("Deleted")

        cv.destroyAllWindows()
        print("Remaining images: ", len(os.listdir(raw_data_path)))
    return None


def emotion_labeller(paths: dict, flag: str) -> None:
    """
    flag: for choosing male || female
    """
    path, positive, neutral, negative = paths[flag]
    items = [ele for ele in os.listdir(path) if ele.endswith(".jpg")]

    for filename in items:
        img = cv.imread(os.path.join(path, filename), cv.IMREAD_COLOR)
        img = cv.resize(img, dsize=(250, 250), interpolation=cv.INTER_LANCZOS4)
        cv.imshow("Pos/ Neu/ Neg", img)

        k = cv.waitKey(0)
        if k == ord('q'):
            # stop programme
            cv.destroyAllWindows()
            break

        elif k == ord('b'):
            shutil.move(os.path.join(path, filename), positive)
            print("Moved to positive")
        elif k == ord('n'):
            shutil.move(os.path.join(path, filename), neutral)
            print("Moved to neutral")
        elif k == ord('m'):
            shutil.move(os.path.join(path, filename), negative)
            print("Moved to negative")
        elif k == ord('j'):
            os.remove(os.path.join(path, filename))
            print("Deleted")

        cv.destroyAllWindows()
        print("Remaining images: ", len(os.listdir(path)) - 3)
    return None


def main() -> None:
    gender_labeller()

    path_dict = {"male": (male_path, male_positive, male_neutral, male_negative),
                 "female": (female_path, female_positive, female_neutral, female_negative)}
    # emotion_labeller(path_dict, flag="male")
    # emotion_labeller(path_dict, flag="female")
    return None


if __name__ == '__main__':
    """
    Split by gender and then by emotion
    Keyboard guildline:
        Gender: "b" for male
                 "m" for female
                 "n" for Delete

        Emotion: "b" for positive
                 "n" for neutral
                 "m" for negative
                 "j" for delete

    """
    main()