from utils.data_processing import torchvision_dataset_align
dataset = torchvision_dataset_align(directory = "./portraits")
dataset.process_and_save
