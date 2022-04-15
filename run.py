from utils.data_processing import zip_processing
from utils.arparse import get_options
import utils.resnet as resnet
import utils.trainning as trainning
from utils.model import Regressor
import os

def main(model):
    opt, _ = get_options()

    if not os.path.exists(opt.datastore_name):
        zip_processing(opt.datastore_name)


    experiment = trainning.experiment(model, opt)
    experiment.fit()



if __name__ == '__main__':
    main(Regressor(3, 100))
