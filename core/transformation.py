import numpy as np

def getRotationMatrix2D(
        center : tuple[int, int] = (0, 0),
        angle : float = 0.0,
        scale : float = 1.0
        ) -> np.ndarray:


    #Angle in degrees to match cv2.getRotationMatrix2D
    theta = np.deg2rad(-angle)  #Negate to match OpenCV's clockwise rotation
    #Finding center
    cx , cy = center

    cos_a = scale * np.cos(theta)
    sin_a = scale * np.sin(theta)

    tx = (1 - cos_a) * cx + sin_a * cy
    ty = (1 - cos_a) * cy - sin_a * cx
    
    M = np.array([
        [cos_a, -sin_a, tx], 
        [sin_a, cos_a, ty]
    ])
    return M

def warpAffine(
        src : np.ndarray,
        M : np.ndarray,
        ) -> np.ndarray:

    # Convert 2x3 to 3x3
    M_full = np.vstack([M, [0, 0, 1]])

    height_input, width_input = src.shape[:2]

    #Affine transforms (and rotation matrices) expect coordinates in the order:
    #[x, y]
    #[col, row]
    #[horizontal, vertical]

    # 4 corners in orders -> [top left, top right, bottom_left, bottom_right]
    corners_input = np.array([
        [0, 0, 1],
        [width_input - 1, 0, 1],
        [0, height_input - 1, 1],
        [width_input - 1, height_input - 1, 1]
    ]).T

    # Get new image corners after transformation
    corners_output = M @ corners_input

    min_x = np.min(corners_output[0])
    max_x = np.max(corners_output[0])
    min_y = np.min(corners_output[1])
    max_y = np.max(corners_output[1])

    width_new = int(np.ceil(max_x - min_x + 1))
    height_new = int(np.ceil(max_y - min_y + 1))

    # Offset to shift image to positive coordinates
    offset_x = -min_x
    offset_y = -min_y

    # Apply offset to transformation matrix
    M[:, 2] += [offset_x, offset_y]

    # Recompute inverse
    M_inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))

    # Initialize output
    rotated_image = np.zeros((height_new, width_new, 3), dtype=np.uint8)

    # Inverse warping
    for y in range(height_new):
        for x in range(width_new):
            src_coords = M_inv @ [x, y, 1]
            col = round(src_coords[0])
            row = round(src_coords[1])

            if 0 <= row < height_input and 0 <= col < width_input:
                rotated_image[y, x, :] = src[row, col, :]

    return rotated_image

