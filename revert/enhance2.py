import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "img3.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    print("Failed to load image.")
    exit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast_enhanced = clahe.apply(gray)

#basic edge detection
edges = cv2.Canny(contrast_enhanced, threshold1=30, threshold2=100)

#thresholding (for hidden patterns/text)
_, thresholded = cv2.threshold(contrast_enhanced, 127, 255, cv2.THRESH_BINARY_INV)

#matlotlib show
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

def add_border_and_label(img, label, font_scale=0.6, text_offset_y=25):
    top_border = 40
    bottom_border = 10
    left_border = 10
    right_border = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0)  
    thickness = 2
    
    bordered = cv2.copyMakeBorder(
        img,
        top_border,
        bottom_border,
        left_border,
        right_border,
        borderType=cv2.BORDER_CONSTANT,
        value=255
    )
    
    cv2.putText(
        bordered,
        label,
        (left_border + 5, text_offset_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return bordered


h, w = contrast_enhanced.shape
if edges.shape != (h, w):
    edges = cv2.resize(edges, (w, h))
if thresholded.shape != (h, w):
    thresholded = cv2.resize(thresholded, (w, h))

contrast_with_border = add_border_and_label(contrast_enhanced, "Contrast Enhanced")
edges_with_border = add_border_and_label(edges, "Edge Detection")
thresholded_with_border = add_border_and_label(thresholded, "Thresholded")

combined = np.hstack([contrast_with_border, edges_with_border, thresholded_with_border])

cv2.imwrite("enhance2_img3.png", combined)
print("enhance2_img.png saved successfully!")