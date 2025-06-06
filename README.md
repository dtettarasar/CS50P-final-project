# AI Art Shield - Image Protection Tool

#### Video Demo: <URL HERE>

#### Description:
AI Art Shield is a command-line tool developed in Python for protecting visual artworks from generative artificial intelligence systems. The project is part of Harvard's CS50P course.

The program offers two main features:

1. **Secure Image** – Takes an image as input, applies a set of invisible perturbations (e.g. subtle noise, hidden watermarking, pixel noise, steganography, etc.) to make the image difficult to exploit by AIs trained to recognize or reproduce artistic styles.

2. **Test Protection** – Allows you to evaluate the effectiveness of the protection by analyzing the recognition of the secure image via a vision model (CLIP type or other local API), and compares the semantics of the description obtained with that of the original image.

This project aims to lay the first brick of a larger system designed to assist artists, publishers, studios and businesses in preserving their intellectual property in the face of predatory AI.

#### Technologies :
- Python 3
- OpenCV / PIL
- Numpy
- argparse / sys

#### Usage :
```bash
# Protect an image
python project.py secure --input image.jpg --output image_protected.jpg

# Test image protection
python project.py test --input image_protected.jpg --reference image.jpg
