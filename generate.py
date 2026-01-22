import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# Config
INPUT_DIR = "input_images"     # 5 images
OUTPUT_DIR = "output_dataset"  # generated images saved here
TARGET_TOTAL = 50              # total images to generate
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Augmentation helpers
def clamp_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)

def random_brightness_contrast(img):
    # brightness: -35..35, contrast: 0.8..1.25
    beta = random.randint(-35, 35)
    alpha = random.uniform(0.8, 1.25)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out

def random_gaussian_noise(img):
    if random.random() < 0.6:
        sigma = random.uniform(3, 12)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return clamp_uint8(out)
    return img

def random_blur(img):
    r = random.random()
    if r < 0.25:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    elif r < 0.35:
        # motion-ish blur
        k = random.choice([5, 7, 9])
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        kernel /= k
        return cv2.filter2D(img, -1, kernel)
    return img

def rotate_image(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return out

def perspective_warp(img, max_shift=0.10):
    """
    Simulates camera viewpoint change (like different angles).
    max_shift = fraction of width/height
    """
    h, w = img.shape[:2]

    # Source corners
    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    dx = w * max_shift
    dy = h * max_shift

    # Randomly move corners inward/outward a bit
    dst = np.float32([
        [random.uniform(-dx, dx), random.uniform(-dy, dy)],
        [w - 1 + random.uniform(-dx, dx), random.uniform(-dy, dy)],
        [w - 1 + random.uniform(-dx, dx), h - 1 + random.uniform(-dy, dy)],
        [random.uniform(-dx, dx), h - 1 + random.uniform(-dy, dy)]
    ])

    P = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, P, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return out

def random_crop_and_resize(img, min_crop=0.85):
    # crop 85%..100% then resize back (zoom effect)
    h, w = img.shape[:2]
    scale = random.uniform(min_crop, 1.0)
    nh, nw = int(h * scale), int(w * scale)

    y1 = random.randint(0, h - nh) if h - nh > 0 else 0
    x1 = random.randint(0, w - nw) if w - nw > 0 else 0

    crop = img[y1:y1 + nh, x1:x1 + nw]
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return out

def augment(img):
    # Order matters a bit: viewpoint/angle-ish transforms first, then photometric
    out = img.copy()

    # 1) Simulate different angles
    out = rotate_image(out, angle_deg=random.uniform(-25, 25))
    out = perspective_warp(out, max_shift=random.uniform(0.04, 0.12))
    out = random_crop_and_resize(out, min_crop=random.uniform(0.82, 0.95))

    # 2) Lighting & camera artifacts
    out = random_brightness_contrast(out)
    out = random_gaussian_noise(out)
    out = random_blur(out)

    return out

# Load input images
image_files = [f for f in os.listdir(INPUT_DIR)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
image_files.sort()

if len(image_files) == 0:
    raise FileNotFoundError(f"No images found in {INPUT_DIR}")

# Generate exactly TARGET_TOTAL images across the available inputs
num_inputs = len(image_files)
per_image_base = TARGET_TOTAL // num_inputs
remainder = TARGET_TOTAL % num_inputs

save_count = 0
idx = 0

print(f"Found {num_inputs} images. Generating {TARGET_TOTAL} total outputs...")

for i, fname in enumerate(image_files):
    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"Skipping unreadable file: {fname}")
        continue

    # how many to generate from this image
    n = per_image_base + (1 if i < remainder else 0)

    for j in tqdm(range(n), desc=f"Augmenting {fname}", leave=False):
        aug = augment(img)

        out_name = f"img_{idx:04d}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, aug, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        idx += 1
        save_count += 1

print(f"Done! Saved {save_count} images to: {OUTPUT_DIR}")