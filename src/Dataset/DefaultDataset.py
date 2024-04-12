import io
import lakefs_spec
import pandas as pd

from PIL import Image
from overrides import override
from torch.utils.data import Dataset
from torchvision.transforms import v2


class DefaultDataset(Dataset):
    __root: str
    __annotation: pd.DataFrame
    __transform: v2.Transform
    __transform: v2.Transform

    def __init__(self, annotation: str, root: str, transform=None, target_transform=None):
        self.__root: str = root
        self.__annotation: pd.DataFrame = pd.read_csv(annotation)

        self.__transform: v2.Transform = transform
        self.__target_transform: v2.Transform = target_transform


    def __len__(self):
        return len(self.__annotation)


    @override
    def __getitem__(self, idx: int):
        path, target = self.__annotation.loc[idx, :]
        # img = self.pil_loader(f"{self.__url}{path}")
        img = self.__pil_loader(path)

        if self.__transform is not None:
            img = self.__transform(img)

        if self.__target_transform is not None:
            target = self.__target_transform(target)
        return img, target

    def __pil_loader(self, path: str) -> Image.Image:
        path = f"{self.__root}{path}"

        if path.startswith("lakefs"):
            fs = lakefs_spec.LakeFSFileSystem()
            img = fs.read_bytes(path=path)
            img = Image.open(io.BytesIO(img))
        else:
            img = Image.open(path)
        return img.convert("RGB")
