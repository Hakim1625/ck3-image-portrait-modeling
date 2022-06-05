from PIL import Image
import zipfile

def zip_processing(dataset_name):
    try:
        with zipfile.ZipFile(f'/datastores/{dataset_name}/{dataset_name}.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
    except:
        None



