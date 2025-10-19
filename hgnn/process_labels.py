from pathlib import Path
import pandas as pd
import numpy as np

load_data_fun = {}


def load_data():
    cell_stage = np.array(pd.read_csv("data/CITE_seq/cell_type.csv", header=None))[0]
    labels = []
    for each in cell_stage:
        if each == "Unclassified":
            labels.append(0)
        elif each == "B_cell":
            labels.append(1)
        elif each == "CD4_T_cell":
            labels.append(2)
        elif each == "CD8_T_cell":
            labels.append(3)
        elif each == "NK":
            labels.append(4)
        elif each == "Monocytes":
            labels.append(5)
        elif each == "DC":
            labels.append(6)
        elif each == "HSC":
            labels.append(7)

    return labels

load_data_fun['CITE_seq'] = load_data

def load_data():
    cell_stage = np.array(pd.read_csv("data/PEA_STA/cell_stage.csv", header=None))[0]
    labels = []
    for each in cell_stage:
        day = each.split("_")[1]
        treat = each.split("_")[2]
        if day == "0h":
            labels.append(0)
        elif day == "6d":
            if treat == "contol":
                labels.append(1)
            elif treat == "BMP4":
                labels.append(2)

    return labels

load_data_fun['PEA_STA'] = load_data

def load_data():
    cell_stage = np.array(pd.read_csv("data/sc_GEM/cell_stage.csv", header=None))[0]
    labels = []
    for each in cell_stage:
        if each == "BJ":
            labels.append("BJ")
        if each == "d8":
            labels.append("d8")
        if each == "d16T-" or each == "d16T+":
            labels.append(each)
        if each == "d24T-" or each == "d24T+":
            labels.append(3)
        if each == "IPS":
            labels.append(4)
        if each == "ES":
            labels.append(5)

    return labels

load_data_fun['sc_GEM'] = load_data

def load_data():
    labels = np.loadtxt("data/sci_CAR/label.txt")
    return labels

load_data_fun['sci_CAR'] = load_data


def load_data():
    cell_stage = np.array(pd.read_csv("data/scNMT/cell_stage.csv", header=None))

    labels = []
    for each in cell_stage:
        if each == "E5.5":
            labels.append(0)
        if each == "E6.5":
            labels.append(1)
        if each == "E7.5":
            labels.append(2)
    labels = np.array(labels)

    return labels

load_data_fun['scNMT'] = load_data


def load_data():
    cell_stage = np.array(pd.read_csv("data/SCoPE2/cell_stage.csv", header=None))[0]
    labels = []
    for each in cell_stage:
        if each == "sc_m0":
            labels.append(0)
        elif each == "sc_u":
            labels.append(1)

    return labels

load_data_fun['SCoPE2'] = load_data


def load_data():
    labels = np.loadtxt("data/SNARE_seq_adult_mouse/label.txt")
    return labels

load_data_fun['SNARE_seq_adult_mouse'] = load_data

def load_data():
    labels = np.loadtxt("data/SNARE_seq_neonatal_mouse/label.txt")

    return labels

load_data_fun['SNARE_seq_neonatal_mouse'] = load_data


def process_labels(save_dir: Path):
    import shutil

    for key, func in load_data_fun.items():
        subdir = save_dir.joinpath(key)
        subdir.mkdir(parents=True, exist_ok=True)
        labels = func()
        if isinstance(labels, list):
            labels = np.array(labels, dtype=int)
        else:
            labels = labels.astype(int)

        f = subdir.joinpath('label.txt')
        np.savetxt(f, labels)
        print('Processed', key)


if __name__ == '__main__':
    process_labels(Path('./data'))
