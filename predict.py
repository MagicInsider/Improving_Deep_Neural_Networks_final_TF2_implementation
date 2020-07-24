import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from utilities import get_root_path

pic_px = 64
pic_py = 64
pic_depth = 3


def load_and_process_image():
    """
    Prompt user to enter filename of the image to classify.
    Should be stored in {program_root}/images directory

    :returns
    image_data -- vector (1, pic_px x pic_py x pic_depth) of resized, flattened and normalized (/255.) image
    PIL_image -- loaded image
    """
    images_path = os.path.join(root_path, 'images/')
    input_message = 'To make a prediction on image stored in ' + images_path + 'input file name, or "STOP" to exit\n>>>'
    image_filename = str(input(input_message))
    if image_filename == 'STOP':
        print('Live long and prosper!')
        sys.exit(0)
    image_path = os.path.join(images_path, image_filename)
    pil_image = Image.open(image_path)

    # cropping pil_image by crop_proportions_ratio (1.00 equals square crop), creating cropped proxy_image
    crop_proportions_ratio = 1.00
    x_base, y_base = pil_image.size[0], pil_image.size[1]
    if x_base > y_base:
        x_crop, y_crop = y_base * crop_proportions_ratio, y_base
    else:
        x_crop, y_crop = x_base, x_base * crop_proportions_ratio

    left = (x_base - x_crop) / 2
    right = left + x_crop
    top = (y_base - y_crop) / 2
    bottom = top + y_crop
    proxy_image = pil_image.crop((left, top, right, bottom))

    # scaling proxy_image down to 64 x 64, flattening and normalizing image data
    proxy_image = proxy_image.resize((pic_px, pic_py))
    image_data = np.array(proxy_image).reshape((1, pic_px * pic_py * pic_depth)) / 255.

    return image_data, pil_image


def mark_image(pil_image, mark):
    """
    Marks source image with prediction
    :parameters
    image -- source image to mark - PIL Image object
    mark -- prediction - nd.array
    :returns
    marked_image - PIL Image object

    """
    base = pil_image.convert('RGBA')
    txt = Image.new("RGBA", base.size, (255, 255, 255, 0))

    # Font size and placement calculation
    x_center, y_center = base.size[0], base.size[1]
    font_size = int(min(base.size) * 0.5)
    x_font, y_font = int(x_center * 0.5 - font_size * 0.33), int(y_center * 0.5 - font_size * 0.66)
    font = ImageFont.truetype(root_path + '/Krungthep.ttf', font_size)

    drawing_context = ImageDraw.Draw(txt)
    drawing_context.text((x_font, y_font), str(mark), font=font, fill=(255, 255, 0, 125))
    marked_image = Image.alpha_composite(base, txt)

    return marked_image


def load_model(filename):
    """
    Loads Keras model
    :parameter
    filename -- Keras model folder name, should be in {program_root/models} directory
    :returns
    model -- Keras model
    """
    model_path = os.path.join(root_path, 'models/', filename)
    model = tf.keras.models.load_model(model_path)
    return model


if __name__ == '__main__':
    root_path = get_root_path()
    model_filename = 'mod-32-64-16-6_0.1-0.04-0.12_0.00074_32_1000_p0.871_r0.842_F0.856'
    Magus = load_model(model_filename)
    print('Predicting by', model_filename, 'model')
    while True:
        image_data, image = load_and_process_image()
        prediction = np.squeeze(np.argmax(Magus.predict(image_data), axis=-1))
        mark_image(image, prediction).show()
