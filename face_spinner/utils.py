import numpy as np
import cv2
import torch


def to_torch_tensor(img):
    """
    Converts input image to PyTorch's tensor

    Args:
        img (numpy.array): Input image

    Returns:
        torch.Tensor -> PyTorch's tensor
    """
    if isinstance(img, list):
        img = np.stack(img, axis=0)
    else:
        img = img[np.newaxis, :, :, :]
    img = img.transpose((0, 3, 1, 2))
    return torch.FloatTensor(img)


def rotate_bound(image, angle):
    """
    Rotate image without clipping

    Args:
        image (numpy.array): Input image
        angle (float): Angle to be rotated

    Returns:
        numpy.array -> Rotated image
    """

    # grab the dimensions of the image and then determine the
    # center
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    new_width = int((width * sin) + (width * cos))
    new_height = int((height * cos) + (height * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (new_width / 2) - center_x
    M[1, 2] += (new_height / 2) - center_y

    # perform the actual rotation and return the image
    rotated_image = cv2.warpAffine(image, M, (new_width, new_height))
    return rotated_image
