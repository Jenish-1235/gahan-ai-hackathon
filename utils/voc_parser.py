# voc_parser.py
import xml.etree.ElementTree as ET
from typing import List, Dict

def parse_voc_annotation(xml_file: str) -> List[Dict]:
    objects = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        cutting = False
        attributes = obj.find('attributes')
        if attributes is not None:
            for attr in attributes.findall('attribute'):
                if attr.find('name').text == 'Cutting':
                    cutting = attr.find('value').text.lower() == 'true'

        objects.append({
            'label': name,
            'bbox': [xmin, ymin, xmax, ymax],
            'cutting': cutting
        })

    return objects