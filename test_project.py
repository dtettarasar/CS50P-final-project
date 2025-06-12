import pytest
import PIL
import numpy as np 
from project import load_image_file, pil_to_numpy, numpy_to_pil

def test_load_image_file():

    img_cs50 = load_image_file("test_files/cs50.jpg")

    assert img_cs50.mode == "RGB"
    assert img_cs50.width == 2048
    assert img_cs50.height == 1366
    assert img_cs50.format == "JPEG"

    img_cookie = load_image_file("test_files/cookie_monster.webp")

    assert img_cookie.mode == "RGB"
    assert img_cookie.width == 1348
    assert img_cookie.height == 1600
    assert img_cookie.format == "WEBP"


def test_load_image_file_errors():

    with pytest.raises(FileNotFoundError):
        assert load_image_file("test_files/wrong_file_name.jpg")

    with pytest.raises(PIL.UnidentifiedImageError):
        assert load_image_file("test_files/test_files.txt")


def test_load_image_file_unexpected_io_error(mocker):
    # Simule une IOError lors de l'appel à Image.open
    # On patche Image.open car c'est la fonction externe qui pourrait lever cette erreur
    mocker.patch('PIL.Image.open', side_effect=IOError("Simulated unexpected I/O error"))

    # On a besoin d'un fichier existant pour que la première vérification (os.path.exists) passe
    # Mais Image.open va ensuite échouer
    dummy_existing_file = "test_files/cs50.jpg" # Ou n'importe quel fichier existant

    with pytest.raises(IOError) as excinfo:
        load_image_file(dummy_existing_file)

    # Tu peux aussi vérifier le message d'erreur si tu le souhaites
    assert "Simulated unexpected I/O error" in str(excinfo.value)
    assert "An unexpected error occurred while opening the image" in str(excinfo.value) # Vérifie le message de ta fonction


def test_load_image_file_other_unexpected_exception(mocker):
    # Simule une autre Exception inattendue lors de l'appel à Image.open
    mocker.patch('PIL.Image.open', side_effect=Exception("Something really bad happened!"))

    dummy_existing_file = "test_files/cs50.jpg"

    with pytest.raises(IOError) as excinfo: # Ton code convertit toutes les 'Exception' en 'IOError'
        load_image_file(dummy_existing_file)

    assert "Something really bad happened!" in str(excinfo.value)
    assert "An unexpected error occurred while opening the image" in str(excinfo.value)

def test_pil_numpy_conversion():

    img_cs50 = load_image_file("test_files/cs50.jpg")

    img_np = pil_to_numpy(img_cs50)

    # type(grocery_list)

    assert type(img_np) == np.ndarray

    assert img_np.ndim == 3 # Doit être un tableau 3D (hauteur, largeur, canaux)
    assert img_np.shape[2] == 3 # Doit avoir 3 canaux (RGB)
    assert img_np.dtype == np.uint8 # Doit être de type uint8 (0-255)

    img_cookie = load_image_file("test_files/cookie_monster.webp")
    img_np_cookie = pil_to_numpy(img_cookie)
    assert type(img_np_cookie) == np.ndarray
    assert img_np_cookie.ndim == 3
    assert img_np_cookie.shape[2] == 3
    assert img_np_cookie.dtype == np.uint8
