import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time

from scoit.cell_analysis import save_scoit_embeddings

def load_data():
    expression_data = np.loadtxt("data/scNMT/expression_data_300.csv")
    promoter_methy_data = np.loadtxt("data/scNMT/promoter_methy_data_300.csv")
    promoter_acc_data = np.loadtxt("data/scNMT/promoter_acc_data_300.csv")

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


    return expression_data, promoter_methy_data, promoter_acc_data, labels


if __name__ == "__main__":

    start_time = time.time()
    expression_data, promoter_methy_data, promoter_acc_data, labels = load_data()
    data = [expression_data, promoter_methy_data, promoter_acc_data]
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)

    sc_model = sc_multi_omics()
    predict_data = sc_model.fit_list_complete(
        data,
        pre_impute=False, # imputation takes a long time.
        dist="gaussian",
        lr=1e-3,
        n_epochs=3000,
        lambda_C_regularizer=0.01,
        lambda_G_regularizer=0.01,
        lambda_O_regularizer=[0.01, 0.01, 0.01],
    )

    save_scoit_embeddings(sc_model, 'scNMT', predict_data)
    print(time.time() - start_time)
