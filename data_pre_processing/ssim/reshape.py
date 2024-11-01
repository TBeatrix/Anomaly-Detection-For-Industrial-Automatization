import os
from PIL import Image
import matplotlib.pyplot as plt


def create_size_histogram(directory):
    sizes = []
          
    for dirpath, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg')):
 

                with Image.open(os.path.join(dirpath, file)) as img:
                    _, height = img.size
                    sizes.append(height)

    # Create a histogram
    plt.hist(sizes, bins=50)
    plt.title('Histogram of Image sizes')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Number of Images')
    #plt.show()
    plt.savefig("image_histogram_normal.png")


def resize_images(source_dir, target_dir, size=(448, 448)):

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg')):
            source_path = os.path.join(source_dir, filename)
            image = Image.open(source_path)
            # Resize the image
            resized_image = image.resize(size,  Image.Resampling.LANCZOS)
            target_path = os.path.join(target_dir, filename)
            resized_image.save(target_path)
         


if __name__ == "__main__":


    directory = "..."
    create_size_histogram(directory)
       