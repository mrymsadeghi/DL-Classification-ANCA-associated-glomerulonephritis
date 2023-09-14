import glob
import cv2 as cv
import numpy as np
import os
from PIL import Image
import PIL.ImageDraw as ImageDraw
import xml.etree.cElementTree as ET
import openslide



mlevel = 1
factor = 2**mlevel

savepath = "/NoBackground"
slidepath = "/Data/WSI"
annotpath = "/Data/Annotations"

slides= sorted(glob.glob(slidepath  + '/**.tiff' , recursive=True))
annotataions=sorted(glob.glob(annotpath + '/**.xml' , recursive=True))



def generate_mask(size, coords, kernel_size, iterations):
    mask_image = Image.new("RGB", (size, size))
    ImageDraw.Draw(mask_image).polygon(coords, fill="#fff")
    mask = np.array(mask_image) / 255
    kernel = np.ones(kernel_size, np.uint8)
    return cv.dilate(mask, kernel, iterations=iterations)


def compute_patch_dimensions(main_dim, secondary_dim, margin_perc):
    main_margin = int(margin_perc * main_dim)
    patch_size = main_dim + 2 * main_margin
    secondary_margin = (patch_size - secondary_dim) // 2
    return patch_size, secondary_margin, main_margin


def extract_patch_and_mask(slide, xmin, ymin, width, height, margin_perc, kernel_size, iterations):
    if height > width:
        patch_size, width_margin, height_margin = compute_patch_dimensions(height, width, margin_perc)
    else:
        patch_size, height_margin, width_margin = compute_patch_dimensions(width, height, margin_perc)
    
    patch_xmin = xmin - width_margin
    patch_ymin = ymin - height_margin
    
    patch_img = np.array(slide.read_region((patch_xmin, patch_ymin), 0, (patch_size, patch_size)).convert("RGB"))
    mask_coords = [(x - xmin + width_margin, y - ymin + height_margin) for x, y in coords]
    mask_img = generate_mask(patch_size, mask_coords, kernel_size, iterations)
    
    return patch_img * mask_img, mask_img



def get_size_and_iterations(dimension, threshold, large_values, small_values):
    if dimension > threshold:
        return large_values
    return small_values


def process_annotation(slide, coords, margin_perc):
    x_coords, y_coords = zip(*coords)
    
    gl_xmin, gl_ymin, gl_width, gl_height = min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)
    
    if gl_height > gl_width:
        size, iterations = get_size_and_iterations(gl_height, 300, [5, 5, 20], [3, 3, 15])
    else:
        size, iterations = get_size_and_iterations(gl_width, 400, [4, 4, 48], [4, 4, 8])

    return extract_patch_and_mask(slide, gl_xmin, gl_ymin, gl_width, gl_height, margin_perc, size, iterations)


def generate_patch(slide_path, xml_path, margin_perc, save_path):
    slide = openslide.OpenSlide(slide_path)
    slide_name = os.path.basename(slide_path).split('.')[0]
    print(slide_name)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for i, annotation in enumerate(root.iter('Annotation'), start=1):
        label = annotation.attrib.get("PartOfGroup")
        coords = [(int(float(coord.attrib.get("X"))), int(float(coord.attrib.get("Y")))) for coord in annotation.find('Coordinates')]
        patch_img, mask_img = process_annotation(slide, coords, margin_perc)
        
        if label:  # Save the images only if a label is specified
            cv.imwrite(os.path.join(save_path, label, f"patch_{slide_name}_{i}_nobg.png"), patch_img[:, :, ::-1])
            cv.imwrite(os.path.join(save_path, label, f"patch_{slide_name}_{i}_mask.png"), mask_img)



for slide in slides:
    slidename= slide.split('\\')[-1]
    annotname = slidename.split('.')[0] + ".xml"
    generate_patch(os.path.join(slidepath, slidename),os.path.join(annotpath,annotname), 0.1)