import os
import sys
sys.path.append('.')
from hydroml import dataset as ds

import argparse
import json
import numpy as np

from inference import run

def main():
    parser = argparse.ArgumentParser(
        prog='hydroml',
        description='Run a machine learning model to generate the final-state net proton pseudorapidity distribution give a dataset of initial-state baryon density distribution.',
    )

    parser.add_argument('-d', '--dataset', nargs=1, required=True, help='loads the initial-state dataset')
    args = parser.parse_args()

    try:
        import time

        start = time.time()

        data = np.fromfile(f'{args.dataset[0]}', dtype=np.float32)
        dataset = ds.InferenceDataset(data, None, 141)
        print(dataset.eta)

        run( dataset, "models/baryon_model_19.6gev.pt" )

        end = time.time()
        print(f"Run time: {end - start} seconds")
    except FileNotFoundError as e:
        print('The file could not be found.. \n')
        print(e)
        parser.print_help()

if __name__ == "__main__":
    main()