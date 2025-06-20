import os
import pytest

import PIL
from PIL import Image, UnidentifiedImageError
from PIL.JpegImagePlugin import JpegImageFile
from PIL.WebPImagePlugin import WebPImageFile

import numpy as np 
from project import load_image_file, pil_to_numpy, numpy_to_pil, apply_dct_protection
from project import _apply_dct_watermark_to_channel
from project import calculate_image_metrics

def test_load_image_file():

    img_cs50 = load_image_file("test_files/cs50.jpg")

    # Test la classe abstraite
    assert isinstance(img_cs50, Image.Image)

    # Test la sous classe de la variable
    assert type(img_cs50) == JpegImageFile

    assert img_cs50.mode == "RGB"
    assert img_cs50.width == 2048
    assert img_cs50.height == 1366
    assert img_cs50.format == "JPEG"

    img_cookie = load_image_file("test_files/cookie_monster.webp")

    assert isinstance(img_cookie, Image.Image)

    assert type(img_cookie) == WebPImageFile

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

    img_cs50_pil_original = load_image_file("test_files/cs50.jpg")
    img_np_cs50 = pil_to_numpy(img_cs50_pil_original)

    assert type(img_np_cs50) == np.ndarray
    assert img_np_cs50.ndim == 3 # Doit être un tableau 3D (hauteur, largeur, canaux)
    assert img_np_cs50.shape[2] == 3 # Doit avoir 3 canaux (RGB)
    assert img_np_cs50.dtype == np.uint8 # Doit être de type uint8 (0-255)


    img_cookie_pil_original = load_image_file("test_files/cookie_monster.webp")
    img_np_cookie = pil_to_numpy(img_cookie_pil_original)

    assert type(img_np_cookie) == np.ndarray
    assert img_np_cookie.ndim == 3
    assert img_np_cookie.shape[2] == 3
    assert img_np_cookie.dtype == np.uint8

    # test conversion back from numpy
    img_cs50_from_numpy = numpy_to_pil(img_np_cs50)

    assert type(img_cs50_from_numpy) == Image.Image # Vérifie que c'est bien un objet PIL Image
    assert img_cs50_from_numpy.mode == "RGB"
    assert img_cs50_from_numpy.width == img_cs50_pil_original.width
    assert img_cs50_from_numpy.height == img_cs50_pil_original.height
    # No need to test the format here, as it remains a PIL object in memory. format is not defined yet, it will be defined during saving process

    img_cookie_from_numpy = numpy_to_pil(img_np_cookie)
    assert type(img_cookie_from_numpy) == Image.Image
    assert img_cookie_from_numpy.mode == "RGB"
    assert img_cookie_from_numpy.width == img_cookie_pil_original.width
    assert img_cookie_from_numpy.height == img_cookie_pil_original.height

    # Vérifier si les valeurs des pixels sont les mêmes après l'aller-retour
    # La conversion PIL -> NumPy -> PIL devrait préserver les valeurs des pixels.
    # Quelques différences mineures peuvent survenir en raison des conversions de type (float vs uint8),
    # mais elles devraient être négligeables (proches de 0).
    # Convertir les deux en float pour la comparaison.
    reconverted_np_cs50 = np.array(img_cs50_from_numpy)
    assert np.allclose(img_np_cs50.astype(float), reconverted_np_cs50.astype(float), atol=1) # atol pour tolérance absolue

    reconverted_np_cookie = np.array(img_cookie_from_numpy)
    assert np.allclose(img_np_cookie.astype(float), reconverted_np_cookie.astype(float), atol=1)

def test_apply_dct_protection_function():

    ...

# --- Fixture pour préparer les données du canal de test ---
# Utiliser une fixture permet de réutiliser les mêmes données préparées pour plusieurs tests
@pytest.fixture(scope="module")
def sample_channel_data():
    """
    Fournit un canal NumPy 2D (float) à partir d'une image pour les tests.
    """
    img_pil = load_image_file("test_files/cs50.jpg")
    img_np = pil_to_numpy(img_pil)
    # Prendre le canal rouge (index 0) et le convertir en float pour qu'il corresponde
    # au type attendu par la fonction (car elle soustrait 128.0)
    return img_np[:, :, 0].astype(float)


# --- Tests pour _apply_dct_watermark_to_channel ---

def test_dct_watermark_output_properties(sample_channel_data):
    """
    Vérifie les propriétés de base du canal après application du filigrane DCT.
    """
    original_channel = sample_channel_data
    strength = 5.0
    seed = 42

    watermarked_channel = _apply_dct_watermark_to_channel(
        original_channel.copy(), strength, seed
    )

    # 1. Vérifier le type de données
    assert watermarked_channel.dtype == np.uint8

    # 2. Vérifier les dimensions
    assert watermarked_channel.shape == original_channel.shape

    # 3. Vérifier que les valeurs de pixels sont dans la plage [0, 255]
    assert np.min(watermarked_channel) >= 0
    assert np.max(watermarked_channel) <= 255


def test_dct_watermark_introduces_change(sample_channel_data):
    """
    Vérifie qu'un filigrane avec strength > 0 introduit bien une modification.
    """
    original_channel = sample_channel_data
    strength = 5.0 # Une force non nulle
    seed = 42

    watermarked_channel = _apply_dct_watermark_to_channel(
        original_channel.copy(), strength, seed
    )

    # Convertir l'original en uint8 pour une comparaison plus directe si allclose échoue
    # Ou comparer les deux en float si tu veux être très précis sur l'algorithme lui-même
    # avant le np.clip et astype(uint8) final.
    # Pour le test final de la fonction, on compare les uint8 résultants.
    
    # Il DOIT y avoir une différence significative
    # np.array_equal est strict, allclose est tolérant.
    # On s'attend à ce que ce ne soit PAS égal
    assert not np.array_equal(original_channel.astype(np.uint8), watermarked_channel)

    # Vérifier une différence moyenne perceptible
    # La différence absolue moyenne entre les pixels originaux et watermarked doit être > un certain seuil
    mean_diff = np.mean(np.abs(original_channel - watermarked_channel.astype(float)))
    # Le seuil (ici 0.5) est à ajuster en fonction de ta strength et de ce qui est "perceptible"
    assert mean_diff > 0.5 # Avec strength=5, la différence devrait être notable


def test_dct_watermark_no_change_with_zero_strength(sample_channel_data):
    """
    Vérifie qu'un filigrane avec strength = 0 n'introduit pratiquement aucun changement.
    """
    original_channel = sample_channel_data
    strength = 0.0 # Force nulle
    seed = 42

    watermarked_channel = _apply_dct_watermark_to_channel(
        original_channel.copy(), strength, seed
    )
    
    # Il devrait y avoir des différences MINIMES dues aux conversions float/uint8 et arrondis
    # np.allclose est parfait ici pour gérer ces petites tolérances
    assert np.allclose(original_channel.astype(np.uint8), watermarked_channel, atol=1)
    # atol=1 signifie une tolérance absolue d'une unité sur les valeurs de pixel (0-255).
    # Cela couvre les arrondis légers qui peuvent survenir.


def test_dct_watermark_reproducibility(sample_channel_data):
    """
    Vérifie que le filigrane est reproductible avec la même graine et la même force.
    """
    original_channel = sample_channel_data
    strength = 7.0
    seed = 123

    watermarked_channel_1 = _apply_dct_watermark_to_channel(
        original_channel.copy(), strength, seed
    )
    watermarked_channel_2 = _apply_dct_watermark_to_channel(
        original_channel.copy(), strength, seed # Même force et même graine
    )

    # Les deux résultats doivent être identiques pixel par pixel
    assert np.array_equal(watermarked_channel_1, watermarked_channel_2)


def test_dct_watermark_different_seed_different_result(sample_channel_data):
    """
    Vérifie que des graines différentes produisent des résultats différents.
    """
    original_channel = sample_channel_data
    strength = 7.0
    seed1 = 123
    seed2 = 456 # Graine différente

    watermarked_channel_1 = _apply_dct_watermark_to_channel(
        original_channel.copy(), strength, seed1
    )
    watermarked_channel_2 = _apply_dct_watermark_to_channel(
        original_channel.copy(), strength, seed2
    )

    # Les deux résultats ne devraient PAS être identiques (sauf cas rarissimes de coïncidence de bruit)
    assert not np.array_equal(watermarked_channel_1, watermarked_channel_2)


def test_dct_watermark_strength_scaling(sample_channel_data):
    """
    Vérifie que l'augmentation de la force augmente l'ampleur du changement.
    """
    original_channel = sample_channel_data.astype(float) # Garder en float pour plus de précision

    seed = 42

    # Force faible
    watermarked_low_strength = _apply_dct_watermark_to_channel(
        original_channel.copy(), strength=1.0, seed_value=seed
    )
    diff_low = np.mean(np.abs(original_channel - watermarked_low_strength.astype(float)))

    # Force élevée
    watermarked_high_strength = _apply_dct_watermark_to_channel(
        original_channel.copy(), strength=10.0, seed_value=seed
    )
    diff_high = np.mean(np.abs(original_channel - watermarked_high_strength.astype(float)))

    # La différence avec une force élevée doit être significativement plus grande
    assert diff_high > diff_low * 2 # Par exemple, au moins deux fois plus grande, à ajuster


def test_dct_watermark_robustness_to_input_dtype(sample_channel_data):
    """
    Vérifie que la fonction gère correctement un input de type uint8 si on la modifie pour l'accepter,
    ou qu'elle produit une erreur si elle attend float et reçoit uint8.
    (Actuellement, elle attend float car tu fais `channel_data.astype(float)` dans `apply_dct_protection`)
    """
    # Si ta fonction _apply_dct_watermark_to_channel attend spécifiquement des floats,
    # assure-toi de lui passer des floats.
    # Le test ci-dessous simule une erreur si tu lui passes un uint8 alors qu'elle attend float
    # et fait des opérations qui nécessitent float (ex: soustraction de float, multiplication).

    # Si la fonction était censée pouvoir gérer les uint8 directement:
    # watermarked_channel = _apply_dct_watermark_to_channel(
    #     sample_channel_data.astype(np.uint8), strength=5.0, seed_value=42
    # )
    # assert watermarked_channel.dtype == np.uint8

    # Si tu as une validation interne qui lève une erreur, teste-la.
    # Ou juste assure-toi que le float est passé depuis la fonction appelante.
    # Ici, la fixture `sample_channel_data` fournit déjà un float, ce qui est correct.

    # A noter qu'actuellement apply_dct_protection() appelle _apply_dct_watermark_to_channel() en passant bien la channel data comme float.
    # Donc pas besoin de gérer un contrôle du type de l'input pour s'assurer qu'on traite bien la channel data en séquence de float et non d'int.
    # Mais si jamais on doit appeler _apply_dct_watermark_to_channel() ailleurs que dans cette fonction :
    # Mieux vaut mettre à jour _apply_dct_watermark_to_channel() avec un block try except pour vérifier que l'on passe en entrée un channel data contenant des floats

    pass # Pas besoin d'un test spécifique si la fonction appelante garantit le type float

def test_calculate_metrics_file_not_found_protected():
    """
    Vérifie que la fonction lève FileNotFoundError avec le message correct, si l'image protégée n'existe pas.
    """
    non_existent_protected_path = "test_files/non_existent_protected.jpg"
    existing_original_path = "test_files/cs50.jpg" # Assure-toi que ce fichier existe pour le test

    # On s'attend à FileNotFoundError et on vérifie une partie du message.
    # /!\ Le message exact de FileNotFoundError peut varier légèrement selon l'OS ou la version de Python,
    # Donc utiliser une regex plus flexible est souvent plus sûr.
    # Ici, on va cibler le nom du fichier.

    with pytest.raises(FileNotFoundError, match=f".*{non_existent_protected_path}.*"):
        calculate_image_metrics(non_existent_protected_path, existing_original_path)


def test_calculate_metrics_file_not_found_original():
    """
    Vérifie que la fonction lève FileNotFoundError avec le message correct, si l'image originale n'existe pas.
    """
    # Crée un fichier "protégé" temporaire qui existe pour ce test
    temp_protected_path = "test_files/temp_protected_for_test.jpg"
    Image.new('RGB', (10, 10)).save(temp_protected_path) # Crée une petite image bidon

    non_existent_original_path = "test_files/non_existent_original.png"

    # On s'attend à FileNotFoundError et on vérifie le nom du fichier original dans le message.
    with pytest.raises(FileNotFoundError, match=f".*{non_existent_original_path}.*"):
        calculate_image_metrics(temp_protected_path, non_existent_original_path)

    os.remove(temp_protected_path) # Nettoyage


def test_calculate_metrics_invalid_image_format_protected():
    """
    Vérifie que la fonction lève UnidentifiedImageError avec le message correct
    pour un format invalide de l'image protégée.
    """
    invalid_image_path = "test_files/not_an_image.txt"
    with open(invalid_image_path, "w") as f:
        f.write("Ceci n'est pas une image, c'est du texte.")
    
    existing_original_path = "test_files/cs50.jpg"

    # Le message de UnidentifiedImageError est souvent générique, comme "cannot identify image file".
    # Utilise une regex qui correspond à ce type de message.
    with pytest.raises(UnidentifiedImageError, match=r".*Unable to identify or open image file 'test_files/not_an_image.txt'*"):
        calculate_image_metrics(invalid_image_path, existing_original_path)

    os.remove(invalid_image_path)

def test_calculate_metrics_invalid_image_format_original():
    """
    Vérifie que la fonction lève UnidentifiedImageError avec le message correct
    pour un format invalide de l'image originale.
    """
    protected_image_path = "test_files/cs50.jpg"
    invalid_original_path = "test_files/not_an_original_image.doc"
    with open(invalid_original_path, "w") as f:
        f.write("Doc doc doc doc.")

    # On s'attend précisément à une UnidentifiedImageError
    with pytest.raises(UnidentifiedImageError, match=r".*Unable to identify or open image file 'test_files/not_an_original_image.doc'*"):
        calculate_image_metrics(protected_image_path, invalid_original_path)

    os.remove(invalid_original_path) # Nettoyage

# todo : test pour ValueError en cas de dimensions différentes entre les deux images.

def test_calculate_metrics_different_dimensions():

    """
    Vérifie que la fonction retourne bien une ValueError si les deux images n'ont pas la même dimension
    """

    protected_image_path = "test_files/cs50.jpg"
    original_image_path = "test_files/cookie_monster.webp"

    with pytest.raises(ValueError, match="^Protected and original images must have the same dimensions for comparison.$"):
        calculate_image_metrics(protected_image_path, original_image_path)


def test_calculate_matrics_same_image():

    """
    Vérifie que l'on a bien les résultats attendus si l'on passe exactement le même fichier, en tant qu'image original et image protégé
    """

    protected_image_path = "test_files/cs50.jpg"
    original_image_path = "test_files/cs50.jpg"

    result = calculate_image_metrics(protected_image_path, original_image_path)

    assert result["mse"] == np.float32(0.0)
    assert result["psnr"] == float('inf')
     