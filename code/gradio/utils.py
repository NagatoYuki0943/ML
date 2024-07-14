import os
import hashlib
from PIL import Image
import requests


def hash_image(image: Image.Image | str) -> str:
    """支持 图片路径/url 和 PIL.Image.Image"""
    sha256 = hashlib.sha256()

    if isinstance(image, Image.Image):
        sha256.update(image.tobytes())
        return sha256.hexdigest()

    elif isinstance(image, str):
        url_or_path = image
        if url_or_path.startswith('http'):
            image = Image.open(requests.get(url_or_path, stream=True).raw).convert('RGB')
            sha256.update(image.tobytes())
            return sha256.hexdigest()

        elif os.path.exists(url_or_path):
            with open(url_or_path, 'rb') as f:
                while True:
                    data = f.read(1024)
                    if not data:
                        break
                    sha256.update(data)
            return sha256.hexdigest()

        else:
            raise ValueError(
                f"could not get image from {url_or_path}")

    else:
        raise ValueError(
            'image should be a str(url/path) or PIL.Image.Image')


def test_hash_image():
    image_path = "images/0001.jpg"
    image = Image.open(image_path)
    image_url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
    print(hash_image(image_path))
    print(hash_image(image))
    print(hash_image(image_url))


if __name__ == '__main__':
    test_hash_image()
