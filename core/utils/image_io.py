import cv2

def load_image(path, color=True):
    """Load an image from a path."""
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE

    img = cv2.imread(path, flag)

    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return img

def save_image(path, img):
    """Save an image to a file."""
    cv2.imwrite(path, img)

