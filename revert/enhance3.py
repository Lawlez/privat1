import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "img4.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# extract LSB pattern
lsb_noise = gray_image % 2
lsb_noise_8u = (lsb_noise * 255).astype(np.uint8) # convert to 8-bit for OpenCV

#apply frequency domain analysis using Discrete Fourier Transform (DFT)
dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

fig, axs = plt.subplots(1, 2, figsize=(15, 6))
axs[0].imshow(lsb_noise_8u, cmap='gray')
axs[0].set_title("Least Significant Bit (LSB) Pattern")
axs[0].axis('off')

axs[1].imshow(magnitude_spectrum, cmap='gray')
axs[1].set_title("Frequency Spectrum Analysis")
axs[1].axis('off')

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


h, w = lsb_noise_8u.shape
if magnitude_spectrum.shape != (h, w):
    magnitude_spectrum = cv2.resize(magnitude_spectrum, (w, h))
    
lsb_with_border = add_border_and_label(lsb_noise_8u, "Least Significant Bit (LSB) Pattern")
fsa_with_border = add_border_and_label(magnitude_spectrum, "Frequency Spectrum Analysis")

combined = np.hstack([lsb_with_border, fsa_with_border])

cv2.imwrite("enhance3_img4.png", combined)
print("enhance3_img.png saved successfully!")