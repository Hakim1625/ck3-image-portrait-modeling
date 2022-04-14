from PIL import Image

def load_img(path):
    return Image.open(path)

from torchvision import transforms, datasets

import os
import shutil 

def zip_processing(dataset_name):
    os.system(f'unzip ~/datastores/{dataset_name}/{dataset_name}.zip -d ./')

def clean_dataset(path='C:/Users/hakim/Dropbox/CK3/Data'):
    scan = os.listdir(path)
    for dir in scan:
        if len(os.listdir(f'{path}/{dir}')) > 4:
            shutil.rmtree(f'{path}/{dir}')

def get_dataset_paths(path, extension='jpg'):
    imgs = []
    scan = os.listdir(path)

    for i, dir in enumerate(scan):
        images = os.listdir(f'{path}/{dir}')
        images = [image for image in images if extension in image]
        for image in images: imgs.append((i, f'{path}/{dir}/{image}'))

    return imgs

import utils.deepface as deepface

def create_align_dataset(path, images):

    parameters = deepface.parameters

    if not os.path.exists(path):
            os.makedirs(path)

    for i, pair in enumerate(images):
        n, image = pair
        newpath = f'{path}/portraits/{n}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        try:
            deepface.face_detect_img(image, f'{newpath}/{i}.png', parameters['size'], parameters['backend'])
        except:
            continue

from utils.dna_parser import dna_to_array
import torch

def save_tensors(data_path, output_path):
    pairs = get_dataset_paths(data_path, extension='txt')
    paths = [path for _, path in pairs]

    if not os.path.exists(f'{output_path}/tensors'):
            os.makedirs(f'{output_path}/tensors')

    dnas = torch.stack([dna_to_array(path) for path in paths])
    torch.save(dnas, f'./{output_path}/tensors/dnas.pt')

    file = lambda path: open(path, "r").read()
    get_gender = lambda file: torch.tensor([0, 1]) if 'female' in file else torch.tensor([1, 0])

    genders = torch.stack([get_gender(file(path)) for path in paths])
    torch.save(genders, f'./{output_path}/tensors/genders.pt')


def process_screenshot_dataset(input_path, output_path):
    clean_dataset(input_path)

    images = get_dataset_paths(input_path)
    create_align_dataset(output_path, images)

    save_tensors(input_path, output_path)