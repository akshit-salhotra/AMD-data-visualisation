import os
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial

def process_image(src_root, dst_root, root, file):
    if not file.lower().endswith('.jpg'):
        return
    
    src_path = os.path.join(root, file)
    img = cv2.imread(src_path, 0)  # Read in grayscale

    if img is None:
        print(f"Warning: Could not read {src_path}")
        return

    # Apply denoising
    img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    # Get relative path and construct destination path
    rel_path = os.path.relpath(src_path, src_root)
    dst_path = os.path.join(dst_root, rel_path)

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Save the denoised image
    cv2.imwrite(dst_path, img)
    print(f"Processed: {src_path} -> {dst_path}")


def copy_jpg_images_parallel(src_root, dst_root):
    tasks = []
    for root, _, files in os.walk(src_root):
        for file in files:
            if file.lower().endswith('.jpg'):
                tasks.append((root, file))

    with Pool(processes=(cpu_count()//2)) as pool:
        func = partial(process_image, src_root, dst_root)
        pool.starmap(func, tasks)


# Example usage
if __name__ == '__main__':
    src_folder = r"d:\cleaning_GUI_annotated_Data\Cirrus_OCT_Imaging_Data"
    dst_folder = r"d:\cleaning_GUI_annotated_Data\denoised_data"

    copy_jpg_images_parallel(src_folder, dst_folder)
