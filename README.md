# AI Art Shield - Image Protection Tool

#### Video Demo: <URL HERE>

#### Description:
**AI Art Shield** is a command-line tool developed in Python for protecting visual artworks from generative artificial intelligence systems. The project is part of Harvard's CS50P course.

The program offers two main features:

1. **Secure Image** – Takes an image as input and applies a set of invisible perturbations (e.g., subtle noise, hidden watermarking, pixel noise, steganography, etc.) to make the image harder to exploit by AIs trained to recognize or reproduce artistic styles.

2. **Test Protection** – Allows you to evaluate the effectiveness of the protection by analyzing the recognition of the secured image via a vision model (CLIP-type or local API), and comparing the semantics of the generated description with that of the original image.

This project lays the foundation for a larger system designed to help artists, publishers, studios, and businesses preserve their intellectual property from predatory AI.

---

#### Technologies:
- Python 3
- Pillow (PIL) / OpenCV
- NumPy
- argparse / sys

---

#### Installation:

1. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

2. Install the required packages:

```
pip install -r requirements.txt

```

#### Usage:

1. Protect an image

```
python project.py secure --input image.jpg --output image_protected.jpg
```

2. Test image protection

```
python project.py verify -p img_protected.jpg -o img_original.jpg
```
