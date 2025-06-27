# AI Art Shield - Image Protection Tool

#### Video Demo: <URL HERE>

#### Description:
**AI Art Shield** is a command-line tool developed in Python for protecting visual artworks from predatory generative artificial intelligence systems (such as midjourney, dall-e, sora, etc...). The project is part of Harvard's CS50P course.

The program offers two main features:

1. **Secure Image** – Takes an image as input and applies a set of invisible perturbations (e.g., subtle noise, hidden watermarking, pixel noise, steganography, etc.) to make the image harder to exploit by AIs trained to recognize or reproduce artistic styles.

2. **Test Protection** – Allows you to evaluate the effectiveness of the protection by analyzing the recognition of the secured image via a vision model (CLIP-type or local API), and comparing the semantics of the generated description with that of the original image.

This project lays the foundation for a larger system designed to help artists, publishers, studios, and businesses preserve their intellectual property from predatory AI.

---

#### Improvements:

This project is a foundational component developed within the CS50P learning program, intended for future implementation in a larger application.

The primary goal for the future is to make these features accessible to any user through a web application (likely developed with Django).

Beyond CS50P, I plan to work on the following key improvements:

- **Code Refactoring**: CS50P's final project requires all code to be in a single file, structured as multiple functions, to meet specific validation rules. While suitable for a learning exercise, this structure isn't ideal for a real-world application. I'll refactor the entire codebase for improved modularity, dividing it into multiple files and/or reimplementing functions within a class structure for better organization and maintainability.

- **Additional Protection Algorithms**: Currently, the project features only DCT (Discrete Cosine Transform) Protection. Future steps include integrating more diverse protection tools, such as Wavelet-based Watermarking, Fourier Transform Watermarking, and Adversarial Perturbation. These protection systems will be continuously upgraded to evolve with AI advancements, ensuring robust defense for illustrators' and photographers' work.

- **Built-in Signature System**: A crucial future feature will be the ability to add an invisible "signature" to artworks, possibly through Invisible QR Code Embedding. This system aims to certify the authenticity of an artwork or picture, verifying its human origin. Ideally, the QR code would store a token signed by the illustrator/photographer's private key.

- **Web3 / Blockchain Integration**: Exploring features related to Web3 and blockchain technologies to enhance provenance, ownership, and unforgeable rights management for digital art.

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
