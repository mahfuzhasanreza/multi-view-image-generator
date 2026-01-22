# Multi-View Image Generator: Synthetic Multi-Angle Dataset Creation

Multi-View Image Generator is a Python-based tool for creating synthetic multi-angle image datasets from a small set of object images.

The tool simulates different camera viewpoints using geometry-aware transformations such as rotation, perspective warping, cropping, and photometric variations (brightness, contrast, noise, and blur). This enables users to significantly expand limited datasets while preserving object structure and visual consistency.

## Setup Instructions for Linux/MacOS

1. Clone the repository:

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




