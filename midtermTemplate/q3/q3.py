import cv2
import matplotlib.pyplot as plt

# Load the color image
image_path = "C:/Users/Acer/OneDrive/Desktop/Study/Y3Sem2/imageProcessing/midterm/tanat.jpg"
color_image = cv2.imread(image_path)
color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# Compute histogram of the color image
hist_color = [cv2.calcHist([color_image], [i], None, [256], [0, 256]) for i in range(3)]

# Plot the histogram of the color image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(hist_color[0], color='red')
plt.title('Histogram - Red Channel')

plt.subplot(1, 3, 2)
plt.plot(hist_color[1], color='green')
plt.title('Histogram - Green Channel')

plt.subplot(1, 3, 3)
plt.plot(hist_color[2], color='blue')
plt.title('Histogram - Blue Channel')

plt.show()

# Perform local histogram equalization of the third quadrant of the color image
height, width, _ = color_image.shape

# Define the coordinates for the third quadrant
start_row = height // 2
start_col = width // 2

# Extract the third quadrant
third_quadrant = color_image[start_row:, :start_col, :]

# Convert the third quadrant to LAB color space
third_quadrant_lab = cv2.cvtColor(third_quadrant, cv2.COLOR_RGB2LAB)

# Equalize the histogram of the L channel in the LAB color space
third_quadrant_lab[:, :, 0] = cv2.equalizeHist(third_quadrant_lab[:, :, 0])

# Convert the third quadrant back to RGB color space
equalized_third_quadrant = cv2.cvtColor(third_quadrant_lab, cv2.COLOR_LAB2RGB)

# Replace the original third quadrant with the equalized one
color_image[start_row:, :start_col, :] = equalized_third_quadrant

# Compute histogram of the equalized color image
hist_equalized_color = [cv2.calcHist([color_image], [i], None, [256], [0, 256]) for i in range(3)]

# Plot the histogram of the equalized color image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(hist_equalized_color[0], color='red')
plt.title('Equalized Histogram - Red Channel')

plt.subplot(1, 3, 2)
plt.plot(hist_equalized_color[1], color='green')
plt.title('Equalized Histogram - Green Channel')

plt.subplot(1, 3, 3)
plt.plot(hist_equalized_color[2], color='blue')
plt.title('Equalized Histogram - Blue Channel')

plt.show()

# Display the original and equalized color images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(color_image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.title('Equalized Image')

plt.show()
