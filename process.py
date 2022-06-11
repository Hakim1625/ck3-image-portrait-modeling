from utils.data_processing import torchvision_dataset_align
import zipfile


try:
    with zipfile.ZipFile('~/datastores/ck3-portraits-aligned/portraits.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
except:
    pass


dataset = torchvision_dataset_align(directory = "./portraits")
dataset.process_and_save
