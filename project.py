from PIL import Image
import os

def main():
    file_name = 'ybear.jpg'
    file_path = os.path.abspath(file_name)
    print(file_path)
    image = Image.open(file_path)
    print(image)


def get_img_file():
    ...


def secure_img():
    ...


def check_img_protection():
    ...


if __name__ == "__main__":
    main()
