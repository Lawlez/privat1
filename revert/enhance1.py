from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

image_path = "img2.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a sharpening and denoise filter to enhance edges
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    denoised = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)

    #apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)

    #upscale again for better visualization
    upscale_factor = 4 
    enhanced_image = cv2.resize(equalized, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

    # Display the enhanced image
    plt.figure(figsize=(8, 8))
    plt.imshow(enhanced_image, cmap='gray')
    plt.axis('off')
    plt.show()
    cv2.imwrite("enhance1_img2.png", enhanced_image)
else:
    print("Failed to load image.")
