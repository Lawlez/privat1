import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "img.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast_enhanced = clahe.apply(gray)

# Apply edge detection to highlight hidden details
edges = cv2.Canny(contrast_enhanced, threshold1=30, threshold2=100)

# Apply thresholding to detect hidden patterns or text
_, thresholded = cv2.threshold(contrast_enhanced, 127, 255, cv2.THRESH_BINARY_INV)

# Display results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(contrast_enhanced, cmap='gray')
axs[0].set_title("Contrast Enhanced")
axs[0].axis('off')

axs[1].imshow(edges, cmap='gray')
axs[1].set_title("Edge Detection")
axs[1].axis('off')

axs[2].imshow(thresholded, cmap='gray')
axs[2].set_title("Thresholded (Possible Hidden Text)")
axs[2].axis('off')

plt.show()
