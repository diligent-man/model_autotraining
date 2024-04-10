import os
import torch
from torchvision.io.image import read_image
from torchvision.transforms import v2, Compose

from pathlib import Path
from tqdm import tqdm
from minio import Minio


def main() -> None:
    # Your code
    client = Minio(
        endpoint="127.0.0.1:9000",
        access_key="trong",
        secret_key="Trong123@",
        secure=False
    )

    filepath = os.path.join(os.getcwd(), "demo_minio")
    os.makedirs(filepath, exist_ok=True)
    bucket_name = "small-celeb-a"
    prefixes = ("train", "test")
    #
    # for prefix in prefixes:
    #     total = len(list(client.list_objects(bucket_name, recursive=True, prefix=prefix)))
    #     for obj in tqdm(client.list_objects(bucket_name, recursive=True, prefix=prefix), total=total):
    #         client.fget_object(bucket_name=bucket_name, object_name=obj.object_name, file_path=os.path.join(filepath, obj.object_name))


    bucket_name = "small-celeb-a"
    client.make_bucket(bucket_name=bucket_name)

    dataset_path = "/home/trong/Downloads/Dataset/small_celeb_A"
    prefix_level1 = ("train", "test")
    prefix_level2 = ("male", "female")
    compose = Compose([
        v2.ToDtype(torch.float),
        v2.ToPILImage()
    ])

    for prefix1 in prefix_level1:
        for prefix2 in prefix_level2:
            total = len(os.listdir(os.path.join(dataset_path, prefix1, prefix2)))

            for img in tqdm(os.listdir(os.path.join(dataset_path, prefix1, prefix2)), total=total):
                object_name = os.path.join(prefix1, prefix2, img)
                file_path = os.path.join(dataset_path, object_name)

                client.fput_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=file_path
                )


    
    return None

if __name__ == '__main__':
    main()