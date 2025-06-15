import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread(r'd:\cleaning_GUI_annotated_Data\Cirrus_OCT_Imaging_Data\000176844\R\20091027\081628\OPT\Carl_Zeiss_Meditec\200X1024X200\Original\B-Scans\000176844_R_20091027_081628_200X1024X200_ORG_IMG_JPG_048.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter in X and Y direction
grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Compute gradient magnitude
grad_magnitude = cv2.magnitude(grad_x, grad_y)

# Normalize for visualization (optional)
grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX)
grad_magnitude = grad_magnitude.astype(np.uint8)

# Plotting
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Sobel X")
plt.imshow(grad_x, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Sobel Edge Magnitude")
plt.imshow(grad_magnitude, cmap='gray')
plt.tight_layout()
plt.show()
