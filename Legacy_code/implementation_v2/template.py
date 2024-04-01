import os
from pathlib import Path
import logging

# logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    # tools: train, eval, inference, visualization
    f"src/tools/main.py",
    f"src/tools/train.py",
    f"src/tools/eval.py",
    f"src/tools/inference.py",
    f"src/tools/visualization.py",
    
    # Saving image from visualization.py
    f"src/report/.gitkeep",
    
    # Model stuff: base, optimizer, loss, ...
    # In case of tweaking model components
    f"src/modelling/__init__.py",

    # Data stuff: preprocessing, loader, custom_dataset, ...
    f"src/data/__init__.py",
    f"src/data/preprocessing.py",
    f"src/data/data_loader.py",
    f"src/data/custom_dataset.py",

    # Misc funcs
    f"src/utils/__init__.py",

    # Config for both training and eval process
    f"src/config.json",

    # In case of trying something
    "Jupyter_notebooks/demo.ipynb",

    # Packages for whole project
    "requirements.txt",

    # "setup.py",
    # "dvc.yaml",
]


def main() -> None:
    for filepath in list_of_files:
        filepath = Path(filepath)  # convert to Windows path
        filedir, filename = os.path.split(filepath)

        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory; {filedir} for the file: {filename}")

        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                pass
                logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filename} is already exists")
    return None


if __name__ == '__main__':
    main()
