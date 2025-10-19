import matplotlib.pyplot as plt
import cv2

def show_image(img, title="Image"):
    """Display an Image using matplotlib."""
    # Convert BGR to RGB if needed
    if (len(img.shape) == 3 and img.shape[2] = 3) :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_two_images(img1, img2, title1='Image1', title2='Image2'):

    """
    Display 2 images side by side for comparison.

    parameters:
    -----------
    img1, img2: ndarray
        Images to display (BGR or grayscale)
    title1, title2 : str
        Titles for each image
    """
    fig, axes = plt.subplot(1, 2, figsize=(12, 6))

    for ax, img, title in zip(axes, [img1, img2], [title1, title2]) :
        # Conver BGR to RGB if image has 3 channels
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title)
        ax.axis('off')

        plt.tight_layout()
        plt.show()



