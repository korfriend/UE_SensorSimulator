import os
import imageio.v2 as imageio


def create_webm_from_images(image_folder, camera_type, output_filename):
    images = []
    print(f"Processing images for {camera_type}...")

    for file_number in range(1401):
        filename = f"{camera_type}_{file_number:06d}.png"
        file_path = os.path.join(image_folder, filename)
        if os.path.exists(file_path):
            images.append(imageio.imread(file_path))
        else:
            print(f"File not found: {file_path}")

    if images:
        print(f"Creating {output_filename}...")
        writer = imageio.get_writer(output_filename, fps=20, codec="libvpx")
        for img in images:
            writer.append_data(img)
        writer.close()
        print(f"{output_filename} created successfully.")
    else:
        print(f"No images found for {camera_type}.")


image_folder = "./labels"
camera_types = ["front", "left", "right", "rear"]

for camera_type in camera_types:
    output_filename = f"semantic_{camera_type}.webm"
    create_webm_from_images(image_folder, camera_type, output_filename)
