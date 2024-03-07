import numpy as np
import cv2
from PIL import Image

def calculate_mse(image1, image2):
    # Ensure this function handles NumPy arrays directly, as it currently does
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    
    mse = np.mean((arr1 - arr2) ** 2)
    return mse

def resize_to_match(image1, image2):
    # Convert NumPy arrays to PIL Images first
    image1 = Image.fromarray(image1)
    image2 = Image.fromarray(image2)
    
    size1 = image1.size
    size2 = image2.size
    
    if size1[0]*size1[1] < size2[0]*size2[1]:
        image1 = image1.resize(size2, Image.Resampling.BILINEAR)
    else:
        image2 = image2.resize(size1, Image.Resampling.BILINEAR)
    
    # Convert back to NumPy arrays to maintain consistency with cv2 image format
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    return image1, image2


def calc_divergence(image, size=9):
    """
    Calculate the average of divergence in size*size boxes throughout the RGB image.
    Divergence is calculated as dR/dx + dG/dy.

    Parameters:
    - image: A numpy array representing the RGB image.
    - size: The size of the box to calculate the divergence over.

    Returns:
    - A float representing the mean of divergence
    - A numpy array representing the divergence for each size*size box.
    """
    # Ensure the image is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array.")

    # Split the image into R, G, B channels
    B, R, G = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Initialize an array to hold the divergence values
    divergence_values = []

    # Calculate dimensions for iteration
    height, width, _ = image.shape
    for y in range(0, height - size + 1):
        for x in range(0, width - size + 1):
            # Extract the current size*size box for each channel
            R_box = R[y:y+size, x:x+size]
            G_box = G[y:y+size, x:x+size]

            # Calculate the partial derivatives
            dR_dx = np.abs(np.diff(R_box, axis=1)).sum() / (size * (size - 1))
            dG_dy = np.abs(np.diff(G_box, axis=0)).sum() / (size * (size - 1))

            # Sum the partial derivatives to get the divergence
            divergence = dR_dx + dG_dy

            # Append the calculated divergence to the list
            divergence_values.append(divergence)

    # Reshape the divergence values to form an array corresponding to the boxes
    num_boxes_y = (height - size) + 1
    num_boxes_x = (width - size) + 1
    divergence_array = np.array(divergence_values).reshape((num_boxes_y, num_boxes_x))

    div_mean = np.mean(divergence_array)

    return div_mean, divergence_array


def calc_curl(image, size=9):
    """
    Calculate the curl in size*size boxes throughout the RGB image.
    Curl is calculated as dR/dx - dG/dy.

    Parameters:
    - image: A numpy array representing the RGB image.
    - size: The size of the box to calculate the curl over.

    Returns:
    - A float represeting the mean of the curl
    - A numpy array representing the curl for each size*size box.
    - A numpy array representing the difference between the B value and the curl value.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array.")
    
    B, R, G = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    height, width = R.shape
    curl_array = np.zeros((height - size + 1, width - size + 1))
    diff_array = np.zeros_like(curl_array)

    for y in range(height - size + 1):
        for x in range(width - size + 1):
            R_box = R[y:y+size, x:x+size]
            G_box = G[y:y+size, x:x+size]

            # Compute partial derivatives
            dR_dx = np.diff(R_box, axis=1).mean()
            dG_dy = np.diff(G_box, axis=0).mean()

            # Calculate curl
            curl = dR_dx - dG_dy
            curl_array[y, x] = curl

            # Calculate average B value in the box
            B_box = B[y:y+size, x:x+size]
            avg_B = B_box.mean()

            # Calculate the difference and assign to diff_array
            diff_array[y, x] = avg_B - curl

    curl_mean = np.mean(curl_array)
    # For regions where diff_array does not cover the entire image,
    # the difference is set to zero, as initialized.

    return curl_mean, curl_array, diff_array

if __name__ == "__main__":
    # Load two images with OpenCV
    image1 = cv2.imread('tensor_LR_set/output_layer_0.png')
    image2 = cv2.imread('tensor_HR_set/output_layer_0.png')
    
    # Ensure images are in the right color format (OpenCV uses BGR, PIL uses RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image1, image2 = resize_to_match(image1, image2)
    
    mse_value = calculate_mse(image1, image2)
    print(f"MSE btw image1 and image 2: {mse_value:.4f}")

    div_mean, div_arr = calc_divergence(image1)
    print("div_mean for image1", round(div_mean, 4))
    div_mean, div_arr = calc_divergence(image2)
    print("div_mean for image2", round(div_mean, 4))