import numpy as np
import csv

from PIL import Image

csv_file = "converted_dataset.csv"

image_data = []
for i in range(10):
    image = Image.open(f"my_numbers/{i}.jpg").convert("L")  # Convert to grayscale
    image = image.resize((28, 28))

    inverted_image = Image.eval(image, lambda x: 252 - x)

    image_array = np.array(inverted_image).flatten()

    image_data.append(image_array)


image_data = np.array(image_data)
image_data = (image_data / 255) * 254
labels = np.arange(10)
dataset = np.column_stack((labels, image_data))


with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(dataset)

print("Dataset conversion complete.")
