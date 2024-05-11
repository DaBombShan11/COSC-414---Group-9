# importing necessary classes and libraries
from PIL import Image 
import time
import os
from skimage import data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
import numpy as np
import concurrent.futures
import psutil

# Directory for where you want downloaded images to be stored
input_directory = "/Users/ashantiboone/Desktop/duplicated_images"
# Directory for where you want processed images to be stored
output_directory = "/Users/ashantiboone/Desktop/duplicated_images-1"
if not os.path.isdir(output_directory):
   os.makedirs(output_directory)

# Function to complete first step in image process -- Convert image 
def convert_image(file_name,idx):
    # creating image object 
    img = Image.open(file_name) 
    
    # using convert method for img1 to create a grayscale version
    img1 = img.convert("L") 
    # using convert method for img2 to create a binary (black and white) version
    img2 = img.convert("1") 
    # img2.show() 
    path = output_directory + "/image_" + str(idx) + ".jpg"
    img2.save(path)
    return path
    
# Function to complete second step in image process -- Resize image
def resize_image(file_name):
    # creating image object 
    img = Image.open(file_name) 
    max_dimension = 1024
    width, height = img.size
    if width > height:
        new_width = max_dimension
        new_height = int((max_dimension / width) * height)
    else:
        new_height = max_dimension
        new_width = int((max_dimension / height) * width)
    resized_img = img.resize((new_width, new_height))
    
    resized_img.save(file_name)

# Funciton to complete third step in image process - segment image
def segment_image(file_name, num_clusters=8):
    # Load the image and immediately resize if needed
    image = cv2.imread(file_name)
    if image.shape[0] > 800:  # Resize large images (height > 800 pixels)
        scale_factor = 800 / image.shape[0]  # Resize to max height of 800 pixels
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    # Convert BGR to RGB (if needed for subsequent operations)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image and convert to float32 for k-means
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Define criteria for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Perform k-means clustering
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to corresponding centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image dimensions
    segmented_image = segmented_image.reshape(image.shape)

    # Convert back to BGR for saving (if needed)
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    # Save the segmented image
    cv2.imwrite(file_name, segmented_image_bgr)

def annotate_image(file_name):
    # Load the image
    image = cv2.imread(file_name)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the center coordinates of the image
    center_x = width // 2
    center_y = height // 2

    # Define the radius of the circle (you can adjust this based on your needs)
    radius = min(center_x, center_y) // 4  # Use 1/4th of the smaller dimension as radius

    # Draw a red circle at the center of the image
    color = (0, 0, 255)  # Red color in BGR format (OpenCV uses BGR)
    thickness = 2  # Thickness of the circle outline
    cv2.circle(image, (center_x, center_y), radius, color, thickness)

    # Save the modified image with the red circle
    cv2.imwrite(file_name, image)

def process_image(file_path, idx):
    print(f"Processing image {idx}...")
    path = convert_image(file_path, idx)
    resize_image(path)
    segment_image(path)
    annotate_image(path)
    print(f"Image {idx} processing completed.")
    
    
def main():
    count = 0
    start = time.perf_counter()
    for filename in os.listdir(input_directory):
        f = os.path.join(input_directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print("On image:", count, f)
            path = convert_image(f, count)
            resize_image(path)
            image = segment_image(path)
            image = annotate_image(path)
            count = count + 1
            
    end = time.perf_counter()
    total = end - start
    total_cpu = psutil.cpu_percent()
    print("Total time to process without threads: ", total)
    print("Total CPU used without threads: ", total_cpu)
    
    count = 0
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx, file_path in enumerate(input_directory):
            futures.append(executor.submit(process_image, file_path, idx))
        concurrent.futures.wait(futures)
    end = time.perf_counter()
    total_time = end - start
    total_cpu = psutil.cpu_percent()
    print(f"Total time to process with multithreading: {total_time:.2f} seconds")
    print("Total CPU used without threads: ", total_cpu)
        
if __name__ == "__main__":
    main()
