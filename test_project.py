from project import load_image_file

def test_load_image_file():

    img_cs50 = load_image_file("test_files/cs50.jpg")

    assert img_cs50.mode == "RGB"
    assert img_cs50.width == 2048
    assert img_cs50.height == 1366
    assert img_cs50.format == "JPEG"


def test_function_2():
    ...


def test_function_n():
    ...