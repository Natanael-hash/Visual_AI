# import os
# from shutil import move
from glob import glob
import pandas as pd
from xml.etree import ElementTree as et
from functools import reduce

xml_list = glob("data_images/*.xml")

def extract_text(filename):
    try:
        tree = et.parse(filename)
        root = tree.getroot()
    except et.ParseError as e:
        print(f"Error parsing XML file {filename}: {e}")
        return []
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return []

    filename_element = root.find("filename")
    image_name = filename_element.text if filename_element is not None else "Unknown"

    size = root.find("size")
    if size is not None:
        width_element = size.find("width")
        height_element = size.find("height")
        width = int(width_element.text) if width_element is not None else 0
        height = int(height_element.text) if height_element is not None else 0
    else:
        width = height = 0

    objects = root.findall("object")
    parser = []

    for obj in objects:
        name = obj.find("name")
        name = name.text if name is not None else "Unknown"

        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = bndbox.find("xmin")
            xmax = bndbox.find("xmax")
            ymin = bndbox.find("ymin")
            ymax = bndbox.find("ymax")

            xmin = int(xmin.text) if xmin is not None else 0
            xmax = int(xmax.text) if xmax is not None else 0
            ymin = int(ymin.text) if ymin is not None else 0
            ymax = int(ymax.text) if ymax is not None else 0
        else:
            xmin = xmax = ymin = ymax = 0

        parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])

    return parser


parser_all = list(map(extract_text, xml_list))
data = reduce(lambda x, y: x + y, parser_all)
df = pd.DataFrame(parser_all, columns=["filename", "width", "height", "name", "xmin", "xmax", "ymin", "ymax"])

cols = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
df[cols] = df[cols].astype(int)

df['center_x'] = ((df['xmin'] + df['xmax']) / 2) / df['width']
df['center_y'] = ((df['ymin'] + df['ymax']) / 2) / df['height']
df['w'] = (df['xmax'] - df['xmin']) / df['width']
df['h'] = (df['ymax'] - df['ymin']) / df['height']

print(len(df['filename'].unique()))