import numpy as np
import sys
import os

def convert(datasets, size=sys.maxsize):
    assert(size >= 1)

    for dataset in datasets:
        print(f"Generating {dataset} dat files\n")
        events = []
        i = 0
        for file in os.listdir(f'./datasets/training/{dataset}'):
            if i >= size:
                break
            # Only register events with baryon_etas
            if file.find('baryon_etas') == -1:
                continue
            # Grab all event numbers
            events.append(file.split('_')[1])
            i += 1

        baryons = []
        protons = []
        latch = False

        for event in events:
            # Eta is the same for the datasets
            eta_baryon, baryon = np.loadtxt(
                f'./datasets/training/{dataset}/event_{event}_net_baryon_etas.txt', unpack=True, dtype=np.float32)
            eta_proton, proton, error = np.loadtxt(
                f'./datasets/training/{dataset}/event_{event}_net_proton_eta.txt', unpack=True, dtype=np.float32)

            if not latch:
                baryons.append(eta_baryon)
                protons.append(eta_proton)
                latch = True

            baryons.append( baryon )
            protons.append( proton )

        np.array(baryons, dtype=np.float32).flatten().tofile(f"{dataset}_netBaryon.dat")
        np.array(protons, dtype=np.float32).flatten().tofile(f"{dataset}_netProton.dat")

def main():
    list = [
        "3DAuAu200_minimumbias_BG16_tune17",
        "NetbaryonDis_OSG3DAuAu19.6_tune18.2_wBulk_22momdeltaf",
        "NetbaryonDis_OSG3DAuAu19.6_tune18.3_wBulk_22momdeltaf",
        "NetbaryonDis_OSG3DAuAu200_tune18.6_wBulk_22momdeltaf_wHBT",
        ]

    convert(list)

if __name__ == "__main__":
    main()
