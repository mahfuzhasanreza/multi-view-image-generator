# Multi-View Image Generator

Multi-View Image Generator is a Python-based tool for creating synthetic multi-angle image datasets from a small set of object images.

The tool simulates different camera viewpoints using geometry-aware transformations such as rotation, perspective warping, cropping, and photometric variations (brightness, contrast, noise, and blur). This enables users to significantly expand limited datasets while preserving object structure and visual consistency.

## Features
- Generate synthetic multi-angle images from limited samples
- Geometry-aware transformations (rotation, perspective warp)
- Photometric augmentations (brightness, contrast, noise, blur)
- Simple, script-based workflow

## Project Structure
The project directory structure is as follows:

```bash
multi-view-image-generator/
├── generate.py
├── input_images
├── output_dataset
└── README.md
```

## Installation (Linux / macOS)

1. Clone and navigate to the repository:

```bash
git clone https://github.com/mahfuzhasanreza/multi-view-image-generator.git
cd multi-view-image-generator
```

2. Create a virtual environment:

```bash
python3 -m venv venv
```

3. Activate the virtual environment:

```bash
source venv/bin/activate
```

4. Install required packages:

```bash
pip install opencv-python numpy tqdm
```

5. Prepare input images:
    - Replace the images in the `input_images` directory with your own object images (5 images recommended).

## Usage

Run the script to generate the multi-view dataset:

```bash
python3 generate.py
```

<br>

## _Author: [Mahfuz Hasan Reza](https://github.com/mahfuzhasanreza/)_
### _Get Connected with [Learn With Mahfuz](https://www.youtube.com/@learn-with-mahfuz)_
  - _Subscribe to my channel on [YouTube](https://www.youtube.com/@learn-with-mahfuz)_
  - _Follow me on [LinkedIn](https://www.linkedin.com/company/learn-with-mahfuz)_
  - _Follow me on [Facebook](https://www.facebook.com/learnwithmahfuzofficial)_
  - _Connect with me on [LinkedIn](https://www.linkedin.com/in/mahfuzhasanreza/)_
