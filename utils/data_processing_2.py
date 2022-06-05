
import dlib
from utils.alignment import align_face
from dataclasses import dataclass, field
from collections.abc import Sequence
from abc import ABC, abstractmethod

import os
import shutil
import imghdr

import numpy as np


class dataset_processor(ABC):
    extensions = ['jpeg', 'jpg', 'png']

    def create_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @property
    def clean_dirs(self) -> None:
        for dir in os.listdir(self.directory):
            sub_dirs = os.listdir(f'{self.directory}/{dir}')
            if (len(sub_dirs) <= 1) or ('.txt' not in ''.join(sub_dirs)):
                shutil.rmtree(f'{self.directory}/{dir}')

        print(f'Directory{self.directory} cleaned')

    @property
    def get_paths(self) -> list[dict[str, list[str]]]:
        sub_dirs = os.listdir(self.directory)
        paths = []

        for dir in sub_dirs:
            sub_path = f'{self.directory}/{dir}'
            files = os.listdir(sub_path)

            images = [f'{sub_path}/{path}' for path in files if (
                imghdr.what(f'{sub_path}/{path}') in dataset_processor.extensions)]
            dna = [f'{sub_path}/{path}' for path in files if '.txt' in path]

            paths.append({'dna': dna[0], 'images': images})

        self.paths = paths

        print('Paths loaded')

    @property
    def process_dataset(self):
        self.create_folder(self.save_directory)
        for i, directory in enumerate(self.paths):

            dir_path = f'{self.save_directory}/portraits#{i}'
            self.create_folder(dir_path)

            images = directory['images']
            dna = directory['dna']

            processed_images = self.process_images(images)

            for ii, image in enumerate(processed_images):
                self.save_processed_image(dir_path, ii, image)

            shutil.copy(dna, f'{dir_path}/dna#{i}.txt')

    @abstractmethod
    def process_images(self, images):
        pass

    @abstractmethod
    def save_processed_image(self, path, index, image):
        pass


@dataclass
class dataset(dataset_processor):
    # paths: [{'dna': 'dna_path',
    #         'images': ['image_path_1', ...]
    #       }, ...]

    paths: list[dict[str, list[str]]] = field(init=True, default=None)
    directory: str = field(init=True, default=None)
    save_directory: str = field(default=None)

    def __post_init__(self):
        if self.directory is not None:
            self.clean_dirs
            self.get_paths

    def __add__(self, other):
        return self.__class__(paths=self.paths + other.paths)


class dataset_align(dataset):

    def __post_init__(self):
        super().__post_init__()
        if self.paths is not None:
            self.predictor = dlib.shape_predictor(
                './utils/dependencies/shape_predictor_68_face_landmarks.dat')

    def align_image(self, image: str):
        return align_face(filepath=image, predictor=self.predictor)

    def process_images(self, images: list[str]) -> list[np.array]:
        processed_images = []

        for image in images:
            try:
                processed_images.append(self.align_image(image))
            except BaseException:
                continue

        return processed_images

    def save_processed_image(self, path: str, index: int, image: PIL.ImageObject):
        image.save(f'{path}/angle#{index}.jpg')
