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

def apply_dct_protection(img_np, strength, verbose_mode):

    logging.info("init apply_dct_protection function")


def check_img_protection():
    ...
    

if __name__ == "__main__":
    main()
