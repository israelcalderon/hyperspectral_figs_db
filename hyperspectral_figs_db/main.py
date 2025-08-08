from typing import Optional
from dotenv import dotenv_values
import cv2
import numpy as np
import pandas as pd
import spectral

import radiometric_corrections
import hs_images


type Coordinate = tuple[int, int]


def mask_image_coordinates(mask_image_path: str) -> list[Coordinate]:
    """
    Gets the pixel coordinates in the provided masks

    Note: Expects an image of color masks with black background

    Returns:
        list: A list of tuples with coordinates in (y, x) or (row, col)
    """
    image = cv2.imread(mask_image_path)
    black = np.array([0, 0, 0])  # This is the only color to filter

    if image is None:
        raise Exception(f'Not image found in {mask_image_path}')
    
    coordinates = np.where(np.all(image != black, axis=-1))
    return list(zip(coordinates[0], coordinates[1]))


def hyperspectral_pixels_from_masks(coordinate_masks: list[Coordinate],
                                    wavelengths: list[float],
                                    cube: np.ndarray) -> pd.DataFrame:
    """
    Extracts from the hs image cube the pixels defined in the coordinate_masks
    
    :param coordinate_masks: List with the positions of each interested pixel
    :param wavelengths: List of wavelengths for each pixel
    :param cube: Hyperspectral image cube
    :return: Data frame with the wavelengts as columns and filtered pixels as rows
    """
    pixels_count = len(coordinate_masks)

    hs_pixels = np.zeros((pixels_count, len(wavelengths)))
    
    for idx, coord in enumerate(coordinate_masks):
        pixel = cube[coord[0], coord[1]] # type: ignore
        hs_pixels[idx] = pixel
    
    return pd.DataFrame(hs_pixels, columns=wavelengths)


def store_dataframe(pixels: pd.DataFrame,
                    class_: int, 
                    path: str,
                    header: bool = False) -> None:
    pixels['class'] = class_
    pixels.to_csv(path, header=header, index=False, mode='a+')


def process_hyperspectral_image(image_data: dict[str, str]) -> None:
    """
    Processes a hyperspectral image based on the provided metadata.

    :param image_data: Dictionary containing paths and class information.
    """
    image_hdr = spectral.open_image(image_data['hs_img_path'])
    wavelengths = image_hdr.bands.centers
    coordinates = mask_image_coordinates(image_data['mask_path'])
    
    hs_cube = radiometric_corrections.black_white_correction(
        image_data['hs_img_path'], 
        image_data['black_ref_path'], 
        image_data['white_ref_path']
    )
    
    pixels_df = hyperspectral_pixels_from_masks(coordinates, wavelengths, hs_cube)
    store_dataframe(pixels_df, image_data['class'], image_data['output_file'], True)


def main() -> None:
    config = dotenv_values()
    images_data = hs_images.get_image_locations(config['IMAGES_DATA_FILE'])  # type: ignore
    
    for image_data in images_data:
        process_hyperspectral_image(image_data)


if __name__ == '__main__':
    main()
