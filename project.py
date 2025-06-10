from PIL import Image
import os
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
        help='Applique des protections anti-IA à une image.',
        description="""
        Cette commande applique un ensemble de techniques de protection
        (comme le filigranage DCT invisible) à une image pour la protéger
        contre l'entraînement des modèles d'IA et l'analyse.
        """
    )

    secure_parser.add_argument(
        '--input', 
        '-i', 
        type=str, 
        required=True, 
        help='Chemin vers l\'image d\'entrée à protéger (ex: image.jpg).'
    )

    secure_parser.add_argument(
        '--output', 
        '-o', 
        type=str, 
        required=True, 
        help='Chemin où sauvegarder l\'image protégée (ex: image_protected.jpg).'
    )

    secure_parser.add_argument(
        '--strength', 
        '-s', 
        type=float, 
        default=5.0, # Valeur par défaut
        help='Force de la protection (valeur flottante, ex: 1.0, 5.0, 10.0). Plus la valeur est élevée, plus la protection est forte mais potentiellement visible.'
    )

    secure_parser.add_argument(
        '--verbose',
        '-v',
        action='store_true', # stocke True si l'argument est présent
        help='Active le mode verbeux pour des informations de débogage supplémentaires.'
    )

    # Analyser les arguments de la ligne de commande
    args = parser.parse_args()

    # Logique principale du programme basée sur les arguments analysés
    if args.verbose:
        # Ici tu configurerais ton logging pour être plus verbeux
        # Par exemple: logging.getLogger().setLevel(logging.DEBUG)
        print("Mode verbeux activé.")
        
    print(f"Commande exécutée : {args.command}")

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
    

if __name__ == "__main__":
    main()
