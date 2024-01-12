import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load grayscale image
image_path = 'C:/Users/Acer/OneDrive/Desktop/Study/Y3Sem2/imageProcessing/midterm/img.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define a filter (kernel/mask)
filter_mask = np.array([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]])

# Apply convolution
img_convolution = cv2.filter2D(img, -1, filter_mask)

# Apply correlation
img_correlation = cv2.filter2D(img, -1, filter_mask, borderType=cv2.BORDER_CONSTANT)
plt.figure(figsize=(15, 5))

# Display the original image, convolution result, and correlation result
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(img_convolution, cmap='gray')
plt.title('Convolution Result'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(img_correlation, cmap='gray')
plt.title('Correlation Result'), plt.xticks([]), plt.yticks([])

plt.show()
