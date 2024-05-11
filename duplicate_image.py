import shutil
import os

# Path to the original image file
original_image_path = "/Users/ashantiboone/Downloads/MorganHacks/GlucodeGroup.JPEG"

# Number of times to copy the image
num_copies = 100

# Create a folder to store the duplicated images
output_folder = "/Users/ashantiboone/Desktop/duplicated_images"
os.makedirs(output_folder, exist_ok=True)

# Copy the image file into the new folder multiple times
for i in range(num_copies):
    try:
        # Generate a unique filename for each copy (e.g., image_001.jpg, image_002.jpg, ...)
        filename = f"image_{i+1:03d}.jpg"
        output_path = os.path.join(output_folder, filename)

        # Copy the original image file to the new filename
        shutil.copyfile(original_image_path, output_path)

        print(f"Image {i+1} duplicated successfully.")
    except Exception as e:
        print(f"Error duplicating image {i+1}: {e}")

print(f"All {num_copies} images duplicated and saved to '{output_folder}' folder.")
