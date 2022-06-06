import pyautogui as auto
import pyperclip
import random

import time
import os


def debug_mouse_position():
    while True:
        os.system('cls')
        loc = auto.position()
        try:
            print(loc)
        except BaseException:
            print(loc)
            break


def normal_distribution(mu, sig, min, max):
    n = random.normalvariate(mu, sig)
    while (n < min) or (n > max) or None:
        n = random.normalvariate(mu, sig)
    return n


def write(path, text):
    with open(path, 'w') as f:
        f.write(text)


class dataset_generator():

    ethnicities = ['african',
                   'arab',
                   'asian',
                   'byzantine',
                   'caucasian',
                   'caucasian_base',
                   'caucasian_blond',
                   'caucasian_brown_hair',
                   'caucasian_dark_hair',
                   'caucasian_ginger',
                   'caucasian_northern_blond',
                   'caucasian_northern_brown_hair',
                   'caucasian_northern_dark_hair',
                   'caucasian_northern_ginger',
                   'circumpolar_blond',
                   'circumpolar_brown_hair',
                   'circumpolar_dark_hair',
                   'east_african',
                   'ethnicity_template',
                   'indian',
                   'mediterranean',
                   'mediterranean_byzantine',
                   'slavic',
                   'slavic_blond',
                   'slavic_brown_hair',
                   'slavic_dark_hair',
                   'slavic_ginger',
                   'slavic_northern_blond',
                   'slavic_northern_brown_hair',
                   'slavic_northern_dark_hair',
                   'slavic_northern_ginger',
                   'south_indian'
                   ]

    animations = ['marshal',
                  'anger',
                  'steward',
                  'shame',
                  'dismissal',
                  'sadness',
                  'chaplain',
                  'beg',
                  'chancellor'
                  ]

    angles = ['camera_head',
              'camera_head+30',
              'camera_head-30'
              ]

    landmarks = {'animations_open': (280, 130),
                 'animations_select': (320, 160),
                 'ethnicity_open': (280, 180),
                 'ethnicity_select': (320, 220),
                 'camera_open': (280, 400),
                 'camera_select': (320, 430),
                 'camera_slider_left': (170, 400),
                 'camera_slider_right': (400, 400),
                 'slider_bounds': (190, 380),
                 'gene_open': (280, 510),
                 'gene_select': (320, 540),
                 'dna_copy_male': (900, 210),
                 'dna_copy_female': (1430, 210),
                 'dna_copy_boy': (900, 725),
                 'dna_copy_girl': (1430, 725),
                 'dna_paste': (1650, 150),
                 'randomize_template': (285, 225),
                 'upper_region': (180, 295, 225, 20),
                 'lower_region': (180, 610, 225, 20),
                 'male_region': (485, 160, 295, 370),
                 'boy_region': (485, 680, 295, 370),
                 'female_region': (1015, 160, 295, 370),
                 'girl_region': (1015, 680, 295, 370),

                 }

    pause = 0.05

    def __init__(self, archetype_size, output_path):

        self.archetypes = []
        self.path = output_path
        self.archetype_size = archetype_size

        auto.PAUSE = dataset_generator.pause
        auto.FAILSAFE = True

    def _generate_archetypes(self):
        del self.archetypes[:]
        for ethnicity in dataset_generator.ethnicities:
            self.archetypes.append({'ethnicity': ethnicity,
                                    'age': normal_distribution(25, 10, 14, 70),
                                    'animation': random.choice(dataset_generator.animations)
                                    }
                                   )

    def _set_age(self, age):
        x, y = auto.locateCenterOnScreen(
            './utils/dependencies/button.png', region=dataset_generator.landmarks['upper_region'], confidence=0.6)

        length = dataset_generator.landmarks['slider_bounds'][1] - \
            dataset_generator.landmarks['slider_bounds'][0] + 25
        x_target = dataset_generator.landmarks['slider_bounds'][0] + (
            (age / 100) * length)

        auto.moveTo(x, y)
        auto.dragTo(x_target, y, button='left', duration=0.5)

    def _set_ethnicity(self, ethnicity):
        x, y = dataset_generator.landmarks['ethnicity_open']

        auto.click(x, y)
        auto.click(x, y)

        for letter in ethnicity:
            auto.press(letter)

        x, y = dataset_generator.landmarks['ethnicity_select']
        auto.click(x, y)

    def _set_animation(self, animation):
        x, y = dataset_generator.landmarks['animations_open']
        auto.click(x, y)
        auto.click(x, y)

        for letter in animation:
            auto.press(letter)

        x, y = dataset_generator.landmarks['animations_select']
        auto.click(x, y)

    def _randomize_clothes(self):
        x, y = dataset_generator.landmarks['gene_open']

        auto.click(x, y)
        auto.click(x, y)

        for letter in 'clothes':
            auto.press(letter)

        x, y = dataset_generator.landmarks['gene_select']

        time.sleep(0.2)
        auto.click(x, y)
        time.sleep(0.2)

        x, y = auto.locateCenterOnScreen(
            './utils/dependencies/button.png', region=dataset_generator.landmarks['lower_region'], confidence=0.6, grayscale=True)

        length = dataset_generator.landmarks['slider_bounds'][1] - \
            dataset_generator.landmarks['slider_bounds'][0] + 25
        x_target = dataset_generator.landmarks['slider_bounds'][0] + (
            random.random() * length)

        auto.moveTo(x, y)
        auto.dragTo(x_target, y, button='left', duration=0.5)

    def _randomize_hair(self):
        x, y = dataset_generator.landmarks['gene_open']

        auto.click(x, y)
        auto.click(x, y)

        for letter in 'hairstyle':
            auto.press(letter)

        x, y = dataset_generator.landmarks['gene_select']

        time.sleep(0.2)
        auto.click(x, y)

        time.sleep(0.2)

        x, y = auto.locateCenterOnScreen(
            './utils/dependencies/button.png', region=dataset_generator.landmarks['lower_region'], confidence=0.6, grayscale=True)

        length = dataset_generator.landmarks['slider_bounds'][1] - \
            dataset_generator.landmarks['slider_bounds'][0] + 25
        x_target = dataset_generator.landmarks['slider_bounds'][0] + (
            random.random() * length)

        auto.moveTo(x, y)
        auto.dragTo(x_target, y, button='left', duration=0.5)

    def _setup_archetype(self, archetype):
        self._set_age(archetype['age'])
        self._set_ethnicity(archetype['ethnicity'])
        self._set_animation(archetype['animation'])
        self._randomize_hair()
        self._randomize_clothes()

    def _switch_angles(self, angle):
        x, y = dataset_generator.landmarks['camera_open']

        auto.click(x, y)
        auto.click(x, y)

        for letter in dataset_generator.angles[angle]:
            auto.press(letter)

        x, y = dataset_generator.landmarks['camera_select']

        time.sleep(0.2)
        auto.click(x, y)

    def _screenshot(self, filename, age, gender):

        if age < 16:
            if gender == 'male':
                auto.screenshot(
                    filename, dataset_generator.landmarks['boy_region'])
            else:
                auto.screenshot(
                    filename, dataset_generator.landmarks['girl_region'])
        else:
            if gender == 'male':
                auto.screenshot(
                    filename, dataset_generator.landmarks['male_region'])
            else:
                auto.screenshot(
                    filename, dataset_generator.landmarks['female_region'])

    def _save_dna(self, path, age, gender):
        if age < 16:
            if gender == 'male':
                x, y = dataset_generator.landmarks['dna_copy_boy']

                auto.click(x, y)
                auto.click(x, y)

                dna = pyperclip.paste()
                write(path, dna)

            else:
                x, y = dataset_generator.landmarks['dna_copy_girl']

                auto.click(x, y)
                auto.click(x, y)

                dna = pyperclip.paste()
                write(path, dna)
        else:
            if gender == 'male':
                x, y = dataset_generator.landmarks['dna_copy_male']

                auto.click(x, y)
                auto.click(x, y)

                dna = pyperclip.paste()
                write(path, dna)
            else:
                x, y = dataset_generator.landmarks['dna_copy_female']

                auto.click(x, y)
                auto.click(x, y)

                dna = pyperclip.paste()
                write(path, dna)

    def _turn(self, path, age):
        x, y = dataset_generator.landmarks['randomize_template']
        auto.click(x, y)
        auto.click(x, y)

        gender = random.choice(['male', 'female'])
        self._save_dna(f'{path}/dna.txt', age, gender)

        for angle in range(3):
            self._switch_angles(angle)
            self._screenshot(f'{path}/{angle}.png', age, gender)

    def _batch(self, archetype, path, n):
        self._setup_archetype(archetype)
        for turn in range(self.archetype_size):
            newpath = f'{path}/{turn+(n*self.archetype_size)}'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            self._turn(newpath, archetype['age'])

    def _generate_dataset(self, n=0):
        self._generate_archetypes()

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        path = f'{self.path}/portrait{n}'

        if not os.path.exists(path):
            os.makedirs(path)

        for i, archetype in enumerate(self.archetypes):
            self._batch(archetype, path, i)


if __name__ == '__main__':
    debug_mouse_position()
