import numpy as np
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.color import rgb2gray
import os

def extract_bit_features(image_path):
    try:
        print(f"Processing image: {image_path}")
        image = imread(image_path)
        if image.ndim == 3 and image.shape[2] == 4:  # Handle RGBA images
            image = image[..., :3]  # Drop the alpha channel
        gray_image = rgb2gray(image)
        gray_image = (gray_image * 255).astype(np.uint8)
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= hist.sum()
        return hist
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_all_image_paths(input_dir):
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def process_images(input_dir, output_file):
    feature_list = []
    image_files = get_all_image_paths(input_dir)
    for idx, image_file in enumerate(image_files):
        print(f"Processing {image_file} ({idx + 1}/{len(image_files)})")
        features = extract_bit_features(image_file)
        if features is not None:
            feature_list.append(features)
    if feature_list:
        np.save(output_file, np.array(feature_list))
        np.save(output_file.replace('.npy', '_paths.npy'), np.array(image_files))
    else:
        print("No features to save.")

if __name__ == "__main__":
    process_images('datasets', 'bit_features.npy')