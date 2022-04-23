from PIL import Image
import zipfile

def load_img(path):
    return Image.open(path)



def loader(path):
    import utils.deepface as deepface   
    return torch.tensor(deepface.face_features(path), dtype=torch.float32)

import os
import shutil 

def zip_processing(dataset_name):
    try:
        with zipfile.ZipFile(f'/datastores/{dataset_name}/{dataset_name}.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
    except:
        None

def remove_empty_dirs(path):
    for dir in os.listdir(path):
        if len(os.listdir(f'{path}/{dir}')) <= 1:
            shutil.rmtree(f'{path}/{dir}')




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

import torch

def create_embedded_dataset(path, images, extension='pt'):

    if not os.path.exists(path):
            os.makedirs(path)

    for i, pair in enumerate(images):
        n, image = pair
        newpath = f'{path}/{n}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        try:
            tensor = loader(image)
            torch.save(tensor, f'{newpath}/{i}.{extension}')
        except:
            continue

def create_dna_dataset(path, dnas, extension='txt'):

    if not os.path.exists(path):
            os.makedirs(path)

    for i, pair in enumerate(dnas):
        n, image = pair
        newpath = f'{path}/{n}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        try:
            os.rename(image, f'{newpath}/{i}.{extension}')
        except:
            continue

        


def process_screenshot_dataset(input_path, output_path):
    clean_dataset(input_path)

    images = get_dataset_paths(input_path)
    create_embedded_dataset(output_path, images)

    dnas = get_dataset_paths(input_path, extension='txt')
    create_dna_dataset(output_path, dnas)

    remove_empty_dirs(output_path)





    