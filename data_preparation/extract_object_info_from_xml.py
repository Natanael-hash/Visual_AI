import os
from shutil import move
from glob import glob
import pandas as pd
from xml.etree import ElementTree as et
from functools import reduce

xml_list = glob("1_datapreparation/data_images/*.xml")


def extract_text(filename):
    tree = et.parse(filename)
    root = tree.getroot()

    image_name = root.find("filename").text
    width = root.find("size").find("width").text
    height = root.find("size").find("height").text

    objects = root.findall("object")
    parser = []

    for object in objects:
        name = object.find("name").text
        bndbox = object.find("bndbox")
        xmin = bndbox.find("xmin").text
        xmax = bndbox.find("xmax").text
        ymin = bndbox.find("ymin").text
        ymax = bndbox.find("ymax").text

        parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])

    return parser


parser_all = list(map(extract_text, xml_list))
data = reduce(lambda x, y: x + y, parser_all)
df = pd.DataFrame(data, columns=["filename", "width", "height", "name", "xmin", "xmax", "ymin", "ymax"])

cols = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
df[cols] = df[cols].astype(int)

df['center_x'] = ((df['xmin'] + df['xmax']) / 2) / df['width']
df['center_y'] = ((df['ymin'] + df['ymax']) / 2) / df['height']
df['w'] = w = (df['xmax'] - df['xmin']) / df['width']
df['h'] = h = (df['ymax'] - df['ymin']) / df['height']

images = df['filename'].unique()
img_df = pd.DataFrame(data, columns=['filename'])
img_train = tuple(img_df.sample(frac=0.8)['filename'])
img_test  = tuple(img_df.query(f"filename not in {img_train}")["filename"])

