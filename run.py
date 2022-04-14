from utils.data_processing import zip_processing
from utils.arparse import get_options
import utils.resnet as resnet
import trainning
import os

def main(model):
    opt = get_options()

    if not os.path.exists(opt.datastore_name):
        zip_processing(opt.datastore_name)


    experiment = trainning.experiment(model, opt)
    experiment.fit()



if __name__ == '__main__':
    main(resnet.ResNet50(3, 100))
