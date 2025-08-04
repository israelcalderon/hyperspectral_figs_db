import csv


def get_image_locations(config_file: str) -> list:
    with open(config_file) as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)
