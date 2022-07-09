"""Build the dataset from dtd files."""

import shutil
from pathlib import Path

DATASET = Path('./data')
OUTPATH = DATASET / 'DTD-80-40-with-label'


def main():
    for idx in range(1, 11):
        train_file = DATASET / 'dtd/labels/train{}.txt'.format(idx)
        test_file = DATASET / 'dtd/labels/test{}.txt'.format(idx)
        val_file = DATASET / 'dtd/labels/val{}.txt'.format(idx)

        train_images = []
        test_images = []
        with open(train_file, 'r') as trf:
            train_images = trf.readlines()

        with open(test_file, 'r') as tef:
            test_images = tef.readlines()

        with open(val_file, 'r') as vlf:
            train_images.extend(vlf.readlines())

        experiment_path = OUTPATH / str(idx)
        experiment_path.mkdir(parents=True, exist_ok=True)

        training_path = experiment_path / 'training'
        training_path.mkdir(parents=True, exist_ok=True)

        testing_path = experiment_path / 'testing'
        testing_path.mkdir(parents=True, exist_ok=True)

        for train_image in train_images:
            src = DATASET / 'dtd/images/' / train_image.strip()
            dst = training_path / train_image.strip()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

        for test_image in test_images:
            src = DATASET / 'dtd/images/' / test_image.strip()
            dst = testing_path / test_image.strip()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)


if __name__ == '__main__':
    main()
