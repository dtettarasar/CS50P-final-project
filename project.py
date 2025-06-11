import PIL
from PIL import Image
import os
import sys
import logging
import argparse
import numpy as np

def main():

    parser = load_parser()

    # Analyser les arguments de la ligne de commande
    args = parser.parse_args()

    # Logique principale du programme basée sur les arguments analysés
    if args.verbose:
        # Ici tu configurerais ton logging pour être plus verbeux
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose mode enabled")
        
    logging.info(f"Command executed : {args.command}")

    if args.command == 'secure':
        logging.info(f"Image to protect : {args.input}")
        logging.info(f"Saved file path : {args.output}")
        logging.info(f"Strength protection : {args.strength}")
        
        secure_img(args.input, args.output, args.strength, args.verbose)

    elif args.command == 'verify':
        print(f"Vérification de l'image : {args.input}")
        if args.output_report:
            print(f"Rapport sauvegardé sous : {args.output_report}")
        if args.strict_mode:
            print("Mode strict de vérification activé.")
            
        # todo
        # appeler la fonction de vérification réelle
        # if os.path.exists(args.input):
        #     # Appeler la fonction de vérification
        #     # result = verify_image_protection(args.input, args.strict_mode)
        #     # Print ou save result
        # else:
        #     print(f"Erreur: Le fichier d'entrée '{args.input}' n'existe pas.")


def load_image_file(img_path):

    """
    Loads an image file using Pillow and performs initial checks.
    Returns a Pillow Image object, or throws an exception on error.
    """

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The input file '{img_path}' was not found.")
    
    try:
        img_pil = Image.open(img_path)
        logging.info(f"Image loaded: '{img_path}', Format: {img_pil.format}, Mode: {img_pil.mode}")

        # --- Débogage de l'objet PIL.Image ---
        logging.debug("--- Attributs de l'objet PIL.Image.Image ---")
        logging.debug(f"Image Mode: {img_pil.mode}")
        logging.debug(f"Image Size (width, height): {img_pil.size}")
        logging.debug(f"Image Width: {img_pil.width}")
        logging.debug(f"Image Height: {img_pil.height}")
        logging.debug(f"Image Format: {img_pil.format}")
        logging.debug(f"Image Bands: {img_pil.getbands()}")
        
        if img_pil.info:
            logging.debug(f"Image Info (metadata): {img_pil.info}")
        else:
            logging.debug("No specific metadata found in img_pil.info.")

        # Convertir l'image en RGB si nécessaire pour assurer 3 canaux pour le traitement DCT
        if img_pil.mode != 'RGB':
            logging.debug(f"Converting image from {img_pil.mode} to RGB mode.")
            img_pil = img_pil.convert('RGB')
        
        return img_pil

    except PIL.UnidentifiedImageError:
        # Cette exception est levée par Pillow si le fichier n'est pas une image valide
        raise PIL.UnidentifiedImageError(f"Unable to identify or open image file '{img_path}'. Check format or corruption.")
    
    except Exception as e:
        # Capture toute autre erreur inattendue lors de l'ouverture du fichier
        raise IOError(f"An unexpected error occurred while opening the image: {e}")


def secure_img(input_path, output_path, strength, verbose_mode):

    logging.info(f"Initiating image protection for: '{input_path}' with strength: {strength}")

    try:
        # 1. Charger l'image PIL
        img_pil = load_image_file(input_path)
        logging.debug(f"Loaded PIL Image: {img_pil}")

        # 2. Convertir l'objet PIL.Image en un tableau NumPy
        img_np = np.array(img_pil)
        logging.debug(f"PIL Image converted to NumPy array with shape: {img_np.shape}")

        # 3. *** APPLIQUER LA PROTECTION DCT ICI ***
        # C'est l'étape où la logique de modification de l'image est exécutée.
        protected_img_np = apply_dct_protection(img_np, strength, verbose_mode)
        logging.debug("DCT protection applied successfully to NumPy array.")

        # --- Reconvertir le tableau NumPy modifié en un objet PIL.Image ---
        protected_img_pil = Image.fromarray(img_np)
        logging.debug("Protected NumPy array converted back to PIL Image.")
        
        # Sauvegarder l'image protégée
        protected_img_pil.save(output_path)
        logging.info(f"Protected image saved successfully to '{output_path}'.")


    except (FileNotFoundError, PIL.UnidentifiedImageError, IOError) as e:
        # Ces exceptions sont levées par load_image_file
        logging.error(f"Error during image processing for secure command: {e}")
        sys.exit(1) # Quitter le programme avec un code d'erreur


def load_parser():

    # argparse settings
    parser = argparse.ArgumentParser(
        description="AI Art Shield is a command-line tool developed in Python for protecting visual artworks from generative artificial intelligence systems. The project is part of Harvard's CS50P course.",
        formatter_class=argparse.RawTextHelpFormatter # Pour garder le formatage des descriptions multilignes
    )

    # Création des sous-commandes (secure, verify)
    subparsers = parser.add_subparsers(
        dest='command', 
        help='available commands', 
        required=True # Rend la sous-commande obligatoire
    )

    # --- Sous-commande 'secure' ---
    secure_parser = subparsers.add_parser(
        'secure', 
        help='Apply anti-IA protections to an image.',
        description="""
        This command applies a set of protection techniques
        (such as invisible DCT watermarking) to an image to protect it
        from AI model training and analysis.
        """
    )

    secure_parser.add_argument(
        '--input', 
        '-i', 
        type=str, 
        required=True, 
        help='Define the path for the input image file to protect (ex: image.jpg).'
    )

    secure_parser.add_argument(
        '--output', 
        '-o', 
        type=str, 
        required=True, 
        help='Define the path for the output image file to save (ex: image_protected.jpg).'
    )

    secure_parser.add_argument(
        '--strength', 
        '-s', 
        type=float, 
        default=5.0, # Valeur par défaut
        help='Protection strength (floating value, e.g. 1.0, 5.0, 10.0). The higher the value, the stronger and more visible the protection.'
    )

    secure_parser.add_argument(
        '--verbose',
        '-v',
        action='store_true', # stocke True si l'argument est présent
        help='Enables verbose mode for additional debugging information.'
    )

    # --- Sous-commande 'verify' (pour plus tard) ---
    verify_parser = subparsers.add_parser(
        'verify', 
        help='Checks whether an image has been protected and/or altered.',
        description="""
        This command analyzes an image to detect the presence of
        protections and verify its integrity via digital signatures or hashes.
        """
    )
    verify_parser.add_argument(
        '--input', 
        '-i', 
        type=str, 
        required=True, 
        help='path for the image file to verify (ex: image_protected.jpg).'
    )
    verify_parser.add_argument(
        '--output-report', # Pourrait générer un rapport
        '-r', 
        type=str, 
        help='path for the report file to save (optionnal).'
    )
    # Exemple d'autres arguments pour la vérification
    verify_parser.add_argument(
        '--strict-mode', 
        action='store_true', 
        help='Activates a strict verification mode to detect the slightest alteration.'
    )

    return parser


def _apply_dct_watermark_to_channel(channel_data, strength, seed_value, verbose_mode=False):
    """
    Prend un seul canal de couleur (par exemple, le canal Rouge) sous forme de tableau NumPy 2D et y applique le filigrane.
    """
    if verbose_mode:
        logging.debug(f"Starting channel processing (shape: {channel_data.shape})")

    # 1. Centrage des données des pixels
    # Les valeurs de pixels sont généralement entre 0 et 255.
    # Pour que la DCT fonctionne mieux (et pour des raisons mathématiques de symétrie),
    # il est courant de centrer les données autour de zéro.
    # On soustrait 128 (qui est environ la moitié de 255).
    centered_channel = channel_data - 128.0
    if verbose_mode:
        logging.debug(f"Centered channel. Min: {np.min(centered_channel)}, Max: {np.max(centered_channel)}")

    # 2. Application de la Transformation Cosinus Discrète (DCT)
    # scipy.fftpack.dct effectue la DCT.
    # Pour une image 2D (qui est ce qu'est un canal), la DCT doit être appliquée deux fois:
    # d'abord sur les lignes, puis sur les colonnes (ou vice-versa).
    # .T (transpose) est utilisé pour appliquer la DCT le long des colonnes après l'avoir appliquée le long des lignes.
    # 'norm='ortho'' assure une transformation orthogonale, ce qui signifie que l'IDCT est simplement l'inverse de la DCT.
    dct_coeffs = dct(dct(centered_channel.T, norm='ortho').T, norm='ortho')
    if verbose_mode:
        logging.debug(f"DCT applied. Coeffs[0,0]: {dct_coeffs[0,0]:.2f}")
    # dct_coeffs contient maintenant les coefficients fréquentiels de l'image.
    # dct_coeffs[0,0] représente la composante DC (la moyenne de l'énergie du canal).

    # 3. Génération du Filigrane (Watermark)
    # np.random.seed(seed_value): Fixe la graine du générateur de nombres aléatoires.
    # C'est CRUCIAL. Si tu appliques le filigrane à nouveau (par exemple, pour la vérification),
    # tu auras besoin de générer EXACTEMENT le même bruit aléatoire. La seed garantit cela.
    # watermark = np.random.normal(0, 2, channel_data.shape): Génère un tableau de bruit
    # aléatoire qui suit une distribution normale (gaussienne) avec une moyenne de 0 et
    # un écart-type de 2. La taille du tableau est la même que celle du canal d'image.
    # Ce bruit sera notre "signature" ou "protection" ajoutée.
    np.random.seed(seed_value)
    watermark = np.random.normal(0, 2, channel_data.shape)
    if verbose_mode:
        logging.debug(f"Watermark generated. Min: {np.min(watermark):.2f}, Max: {np.max(watermark):.2f}")

    # 4. Injection du Filigrane dans les Coefficients DCT
    # C'est l'étape clé du filigranage. Nous ajoutons le bruit généré aux coefficients DCT.
    # La 'strength' est un facteur de mise à l'échelle. Plus la force est élevée,
    # plus le filigrane est prononcé (plus visible, mais aussi plus résistant).
    dct_coeffs += strength * watermark
    if verbose_mode:
        logging.debug(f"Watermark added to DCT coefficients. New Coeffs[0,0]: {dct_coeffs[0,0]:.2f}")

    # 5. Application de la Transformation Cosinus Discrète Inverse (IDCT)
    # Nous utilisons idct pour revenir du domaine fréquentiel au domaine spatial (pixels).
    # C'est l'inverse exact de l'étape 2.
    reconstructed_centered_channel = idct(idct(dct_coeffs.T, norm='ortho').T, norm='ortho')
    if verbose_mode:
        logging.debug(f"IDCT applied.")

    # 6. Dé-centrage et Limitation des valeurs des pixels
    # Nous ajoutons 128.0 pour ramener les valeurs à la plage 0-255.
    # np.clip(..., 0, 255): Les transformations peuvent parfois produire des valeurs
    # en dehors de la plage valide [0, 255]. `np.clip` assure que toutes les valeurs
    # restent dans cette plage en les "coupant" si elles sont trop basses (<0) ou trop hautes (>255).
    # .astype(np.uint8): Convertit le tableau en type entier non signé de 8 bits,
    # ce qui est le format standard pour les pixels d'image (0-255).
    watermarked_channel = np.clip(reconstructed_centered_channel + 128.0, 0, 255).astype(np.uint8)
    if verbose_mode:
        logging.debug(f"Processed and clipped channel. Min: {np.min(watermarked_channel)}, Max: {np.max(watermarked_channel)}")
    
    return watermarked_channel


def apply_dct_protection(img_np, strength, verbose_mode=False):
    """
    Applique un filigrane DCT à tous les canaux d'une image couleur NumPy.
    Prend un NumPy array en entrée et retourne un NumPy array modifié.
    """
    if verbose_mode:
        logging.info(f"Applying DCT protection on all channels with strength={strength}")

    # Vérification des dimensions de l'image
    # img_np.ndim: Nombre de dimensions du tableau (2 pour niveaux de gris, 3 pour couleur).
    # img_np.shape[2]: Taille de la 3ème dimension (nombre de canaux).
    # On s'assure que l'image a au moins 3 canaux (pour RGB/BGR).
    if img_np.ndim < 3 or img_np.shape[2] < 3:
        logging.error("Input image must have at least 3 channels (RGB/BGR) for multi-channel DCT protection.")
        # Retourne une copie si ce n'est pas une image couleur, pour éviter un crash.
        return img_np.copy()

    # Créer une copie de l'image NumPy
    # C'est important ! On ne veut pas modifier l'array original en place si cette fonction est appelée
    # et que l'appelant veut garder l'original inchangé.
    processed_image_np = img_np.copy()

    # Itération sur les canaux (R, G, B)
    # range(3) pour les trois premiers canaux (0, 1, 2) correspondant à R, G, B dans Pillow.
    for i in range(3):
        channel_name = ["Red", "Green", "Blue"][i]
        if verbose_mode:
            logging.info(f"Processing channel: {channel_name}")
        
        # Génération d'une graine (seed) différente pour chaque canal
        # channel_seed = 42 + i: Permet de générer un filigrane aléatoire DIFFERENT pour chaque canal.
        # C'est une bonne pratique pour augmenter la robustesse et la complexité du filigrane.
        # L'utilisation de 42 est juste un nombre arbitraire "magique" souvent utilisé pour les graines.
        channel_seed = 42 + i

        # Extraction du canal et conversion en float
        # processed_image_np[:, :, i]: Sélectionne toutes les lignes, toutes les colonnes,
        # et le i-ème canal. Cela extrait un tableau 2D (le canal).
        # .astype(float): La DCT fonctionne mieux avec des nombres flottants, car les
        # coefficients peuvent être non entiers. On convertit explicitement le canal en float.
        processed_channel = _apply_dct_watermark_to_channel(
            processed_image_np[:, :, i].astype(float),
            strength,
            channel_seed,
            verbose_mode=verbose_mode
        )
        
        # Réassignation du canal traité
        # Le canal modifié est remis à sa place dans le tableau d'image complet.
        processed_image_np[:, :, i] = processed_channel

    if verbose_mode:
        logging.info("DCT protection applied to all channels.")
    
    return processed_image_np
    

if __name__ == "__main__":
    main()
