import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reload the image
image_path = "img.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract LSB pattern
lsb_noise = gray_image % 2

# Apply frequency domain analysis using Discrete Fourier Transform (DFT)
dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Display the LSB pattern and frequency spectrum
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

axs[0].imshow(lsb_noise, cmap='gray')
axs[0].set_title("Least Significant Bit (LSB) Pattern")
axs[0].axis('off')

axs[1].imshow(magnitude_spectrum, cmap='gray')
axs[1].set_title("Frequency Spectrum Analysis")
axs[1].axis('off')

plt.show()
