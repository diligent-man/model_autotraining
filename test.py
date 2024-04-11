# https://pydocs-lakefs.lakefs.io/
import os.path

import lakefs_spec
import fsspec


def main() -> None:
    # Should not pass anything to these class. It still has bug with kwarg
    # Credentials will be read from .lakectl.yaml
    repository_id = "celeb-a"
    branch_id = "main"
    ref = "95e7992e51e5"

    fs = lakefs_spec.LakeFSFileSystem()
    fs.download(lpath=repository_id, rpath=f"lakefs://{repository_id}/{branch_id}/", recursive=True, callback=fsspec.callbacks.TqdmCallback(size=1))
    return None


if __name__ == '__main__':
    main()