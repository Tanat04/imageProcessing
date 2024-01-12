from PIL import Image
import numpy as np

# Load the image
image_path = "C:/Users/Acer/OneDrive/Desktop/Study/Y3Sem2/imageProcessing/midterm/tanat.jpg"
original_image = Image.open(image_path)

# Convert the image to a NumPy array
original_array = np.array(original_image)

# Extract R, G, B components
r_channel = original_array[:, :, 0]
g_channel = original_array[:, :, 1]
b_channel = original_array[:, :, 2]

# Subtract distinct intensities from the maximum pixel intensity
max_intensity = 255  # Assuming 8-bit image

modified_r = max_intensity - r_channel
modified_g = max_intensity - g_channel
modified_b = max_intensity - b_channel

# Create a modified image array
modified_array = np.stack([modified_r, modified_g, modified_b], axis=-1)

# Convert the modified array to Image
modified_image = Image.fromarray(modified_array.astype('uint8'))

# Compute the difference between the original and modified images
difference_array = original_array - modified_array
difference_image = Image.fromarray(np.abs(difference_array).astype('uint8'))

modified_image.save("modified_image.jpg")
difference_image.save("difference_image.jpg")

# Create a blank image to display all three images side by side
combined_width = original_image.width * 3
combined_height = original_image.height
combined_image = Image.new("RGB", (combined_width, combined_height))

# Paste original, modified, and difference images side by side
combined_image.paste(original_image, (0, 0))
combined_image.paste(modified_image, (original_image.width, 0))
combined_image.paste(difference_image, (2 * original_image.width, 0))

# Display the combined image
combined_image.show()

# Save the combined image
combined_image.save("combined_result.jpg")
