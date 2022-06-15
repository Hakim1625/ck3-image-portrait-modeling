from torchvision import datasets
from torchvision.transforms import PILToTensor, ToPILImage
from torch.utils.data import DataLoader, Dataset

from utils.model import model as net
from utils.parser import get_options
from utils.training import experiment
from utils.dna_parser import tensor_ToDna



class dataset(Dataset):
    def __init__(self, dataset_path):
        self.images = datasets.ImageFolder(root=f'./{dataset_path}/', transform=self.transform)
        self.n_samples = len(self.images) 

    def transform(self, image):
        return PILToTensor()(image)/255

    def __getitem__(self, index):
        image, i = self.images[index]
        return image

    def __len__(self):
        return self.n_samples

images = dataset('./predictions/processed')
dataloader = DataLoader(images, batch_size = 1)

parser = get_options()[0]

model = net(3, 256, 229)

run = experiment(model, parser)
predictions = run.predict(dataloader, 'logs/Regressor/version_18/checkpoints/epoch=99-step=71599.ckpt')


for i, prediction in enumerate(predictions):
    tensor_ToDna(prediction[0], f'./predictions/predictions/dna#{i}.txt')
    image = ToPILImage()(images[i])
    image.save(f'./predictions/predictions/image#{i}.png')


