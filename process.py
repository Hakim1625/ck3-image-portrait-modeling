from utils.data_processing import torchvision_dataset_align
import zipfile

with zipfile.ZipFile('/datastores/ck3-portraits/portraits.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

dataset = torchvision_dataset_align(directory = "./portraits")
dataset.num_workers = 1
dataset.process_and_save



