from utils.data_processing import torchvision_dataset_align
import zipfile
import shutil

with zipfile.ZipFile('/datastores/ck3-portraits/portraits.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

dataset = torchvision_dataset_align(directory = "./portraits")
dataset.num_workers = 4
dataset.process_and_save


shutil.make_archive('./', 'zip', './portraits_aligned')
