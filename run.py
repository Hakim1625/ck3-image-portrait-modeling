from utils.model import model as net
from utils.parser import get_options
from utils.training import experiment

parser = get_options()[0]

model = net(3, 256, 229)

run = experiment(model, parser)
run.fit()