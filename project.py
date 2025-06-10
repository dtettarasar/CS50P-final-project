from PIL import Image
import os
import sys
import argparse

def main():

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

    # Analyser les arguments de la ligne de commande
    args = parser.parse_args()

    # Logique principale du programme basée sur les arguments analysés
    if args.verbose:
        # Ici tu configurerais ton logging pour être plus verbeux
        # Par exemple: logging.getLogger().setLevel(logging.DEBUG)
        print("Verbose mode enabled")
        
    print(f"Command executed : {args.command}")

    if args.command == 'secure':
        print(f"Image to protect : {args.input}")
        print(f"Saved file path : {args.output}")
        print(f"Strength protection : {args.strength}")
        
        # todo 
        # appeler la fonction de protection réelle
        # if os.path.exists(args.input):
        #     # Appeler la fonction de protection
        #     # protected_image = apply_dct_watermark_multi_channel(args.input, args.strength)
        #     # Si successful: save protected_image to args.output
        # else:
        #     print(f"Erreur: Le fichier d'entrée '{args.input}' n'existe pas.")

        load_image_file(args.input)

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

    formats_list = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    try:

        img_pil = Image.open(img_path)
        print(img_pil)

    except FileNotFoundError:
        # This exception will be thrown if Image.open() does not find the file
        sys.exit(f'Error: The input file ‘{img_path}’ was not found.')

    except Image.UnidentifiedImageError:
        # Cette exception est levée par Pillow si le fichier n'est pas une image valide
        sys.exit(f"Error: Unable to identify or open image file ‘{img_path}’. Check format or corruption.")


def secure_img(img_file_path, protection_strength):
    ...


def check_img_protection():
    ...
    

if __name__ == "__main__":
    main()
