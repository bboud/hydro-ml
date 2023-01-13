import os
import sys
sys.path.insert(0, os.path.abspath(''))

import argparse
import json

from hydroml.inference import run

def main():

    try:
        f = open('config.json', mode='r')
        config = json.load(f)
        f.close()

        print(config)
    except FileNotFoundError:
        print('Cannot write default config file! You need to create a config.json!')
        return

    parser = argparse.ArgumentParser(
        prog='hydroml',
        description='Run a machine learning model to generate the final-state net proton pseudorapidity distribution give a dataset of initial-state baryon density distribution.',
    )

    parser.add_argument('-d', '--dataset', nargs=1, required=True, help='loads the initial-state dataset')
    args = parser.parse_args()
    try:
        # Abstraction to accommodate different data loading methods.
        run( args.dataset[0], config['gridNx'] )
    except FileNotFoundError:
        print('The file could not be found.. \n')
        parser.print_help()

if __name__ == "__main__":
    main()