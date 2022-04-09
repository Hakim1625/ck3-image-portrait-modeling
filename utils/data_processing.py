import os
import shutil 
import pickle


def dump(path, array):
    with open(path, "wb") as f:
        pickle.dump(array, f)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

from PIL import Image


def load_img(path):
    return Image.open(path)

import torch
import torch.nn as nn
from itertools import combinations, combinations_with_replacement
from torchvision import datasets, transforms, models, utils
from torch.utils.data import Dataset
class portraits_dataset(Dataset):
    def __init__(self, image_tensors, slider_tensors):
        self.inputs = image_tensors
        self.targets = slider_tensors

        self.n_samples = len(self.targets)

        self.combinations = list(combinations_with_replacement(range(3), 2)) + list(combinations(reversed(range(3)), 2))
        self.groups = [[a, b, c] for a, b, c in zip(range(0, self.n_samples, 3), range(1, self.n_samples, 3), range(2, self.n_samples, 3))]


    def __getitem__(self, index):
        return torch.stack([self.get_input(combination, self.groups[int(index/3)]) for combination in self.combinations]), self.targets[index]

    def __len__(self):
        return self.n_samples
    
    def get_input(self, combination, keys):
        get_tensor  =  lambda keys, i: self.inputs[keys[combination[i]]]
        t1, t2 = get_tensor(keys, 0), get_tensor(keys, 1)
        
        return torch.stack([t1, t2])
class mask_rcnn(nn.Module):
    def __init__(self, threshold):
        super(mask_rcnn, self).__init__()

        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to('cuda').eval()
        self.threshold = threshold

    def forward(self, batch):
        with torch.no_grad():
            masks = self.model(batch)
            masks_batch = [mask.get('masks')[0][0] for mask in masks]
            masks_batch = [mask > 0.45 for mask in masks_batch]
            masks_batch = [image*mask for image, mask in zip(batch, masks_batch)]
            return torch.stack(masks_batch)

def transform(img, model=mask_rcnn(0.45)):
    with torch.no_grad():
        img_size = img._size
        if img_size[0] > img_size[1]:
            size = img_size[0]-50
        else:
            size = img_size[1]-50

        ccrop = transforms.CenterCrop(size)
        resize = transforms.Resize(224)
        tensor = transforms.PILToTensor()

        mask = lambda x: torch.squeeze(model(torch.unsqueeze(x/255, dim=0).to('cuda')))
        

        return mask(tensor(resize(ccrop(img))).float())

def clean_dataset(path='C:/Users/hakim/Dropbox/CK3/Data'):
    scan = os.listdir(path)
    for dir in scan:
        if len(os.listdir(f'{path}/{dir}')) > 4:
            shutil.rmtree(f'{path}/{dir}')

def create_processed_folder(path='./portraits'):

    if not os.path.exists(path):
            os.makedirs(path)


    n = 0
    for element in images:
        i = element[1]
        newpath = f'{path}/{i}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        utils.save_image(element[0], f'{newpath}/{n}.jpg')
        n += 1


def process_image_dataset():
    clean_dataset()
    create_processed_folder()




from utils.dna_parser import dna_to_array
root = 'C:/Users/hakim/Dropbox/CK3/Portraits'
root_2 = 'C:/Users/hakim/Dropbox/CK3/Data'
def main():

    dnas = [dna[0] for dna in datasets.DatasetFolder(root_2, loader=dna_to_array, extensions=('txt'))]
    #images = datasets.ImageFolder(root, transform=transforms.ToTensor())


    
    torch.save(dnas, 'dataset_dna')
    #torch.save(images, 'dataset_images')
    

def main2():
    dna = torch.load('dataset_dna')
    images = torch.load('dataset_images')

    print(len(images))
    

if __name__ == "__main__":
    main()

