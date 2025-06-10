from PIL import Image
import os
import argparse

def main():

    arg_settings()

    # file_name = 'ybear.jpg'
    # file_path = os.path.abspath(file_name)
    # print(file_path)
    # image = Image.open(file_path)
    # print(image)


def get_img_file():

    formats_list = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    ...


def secure_img():
    ...


def check_img_protection():
    ...

def arg_settings():

    parser = argparse.ArgumentParser(
        description="AI Art Shield is a command-line tool developed in Python for protecting visual artworks from generative artificial intelligence systems. The project is part of Harvard's CS50P course.",
        formatter_class=argparse.RawTextHelpFormatter # Pour garder le formatage des descriptions multilignes
    )

    parser.print_help()

if __name__ == "__main__":
    main()
