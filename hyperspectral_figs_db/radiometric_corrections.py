import cv2
import numpy as np
import pandas as pd
import spectral


def create_cube_reference(hdr_reference: str, hdr_image: str) -> np.ndarray:
    reference_img = spectral.open_image(hdr_reference)
    num_columns, num_bands = int(reference_img.ncols), int(reference_img.nbands)
    cols_bands = np.zeros((num_columns, num_bands))
    
    img = spectral.open_image(hdr_image)
    num_rows = int(img.nrows)

    for col in range(num_columns):
        for band in range(num_bands):
            cols_bands[col, band] = np.mean(reference_img[:, col, band])
    return np.tile(cols_bands, [num_rows, 1, 1])


def black_white_correction(hdr_image: str,
                           hdr_black_reference: str,
                           hdr_white_reference: str) -> np.ndarray:
    black = create_cube_reference(hdr_black_reference, hdr_image)
    white = create_cube_reference(hdr_white_reference, hdr_image)
    image = spectral.open_image(hdr_image).load()

    return np.subtract(image, black) / np.subtract(white, black)
