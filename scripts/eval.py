"""Evaluate trained models."""

import json
from glob import glob
from pathlib import Path


def avg_accuracy(files: list) -> float:
    """Calculate the average accuracy for all runs based on max accuracy.

    Args:
        files: List of files to evaluate.

    Returns:
        float: Average accuracy.
    """
    accs = []
    for fin in files:
        if Path(fin).is_dir():
            with open(Path(fin) / 'history.json') as fjson:
                history = json.load(fjson)
            accs.append(max(history['val_acc_history']))
    return sum(accs) / len(accs)


def main():
    """Run evaluation."""
    base_path = Path('./models')
    multi_layer = base_path / 'multi_layer'
    single_aux = base_path / 'single_layer_with_aux'

    ml_files = glob(str(multi_layer / '*'))
    sa_files = glob(str(single_aux / '*'))

    print('Multi-layer:', avg_accuracy(ml_files))
    print('Single-layer with aux:', avg_accuracy(sa_files))


if __name__ == '__main__':
    main()
