import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread(r'd:\cleaning_GUI_annotated_Data\Cirrus_OCT_Imaging_Data\000176844\R\20091027\081628\OPT\Carl_Zeiss_Meditec\200X1024X200\Original\B-Scans\000176844_R_20091027_081628_200X1024X200_ORG_IMG_JPG_048.jpg', cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur to reduce noise (recommended before Canny)
blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
# blurred=img

# Apply Canny edge detector
# You can tune the thresholds: lower and upper
# edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
denoised = cv2.medianBlur(img, 5)  # Kernel size: 3, 5, or 7 (odd)
denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
# Show results
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Canny Edges")
plt.imshow(denoised, cmap='gray')

plt.tight_layout()
plt.show()
