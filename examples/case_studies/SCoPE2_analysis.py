import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time

from scoit.cell_analysis import save_scoit_embeddings

def load_data():
    expression_data = np.array(pd.read_csv("data/SCoPE2/expression_data.csv", index_col=0))
    protein_data = np.array(pd.read_csv("data/SCoPE2/protein_data.csv", index_col=0))
    cell_stage = np.array(pd.read_csv("data/SCoPE2/cell_stage.csv", header=None))[0]
    # labels = []
    # for each in cell_stage:
    #     if each == "sc_m0":
    #         labels.append(0)
    #     elif each == "sc_u":
    #         labels.append(1)

    return expression_data, protein_data, cell_stage #labels


if __name__ == "__main__":

    start_time = time.time()
    expression_data, protein_data, labels = load_data()
    data =np.array([expression_data, protein_data])
    print(data.shape)

    sc_model = sc_multi_omics()
    predict_data = sc_model.fit_complete(
        data,
        dist="gaussian",
        lr=1e-3,
        n_epochs=5000,
        lambda_C_regularizer=0.01,
        lambda_G_regularizer=0.01,
        lambda_O_regularizer=[0.01, 0.01],
        lambda_OC_regularizer=[1, 1],
        lambda_OG_regularizer=[1, 1],
    )

    save_scoit_embeddings(sc_model, 'SCoPE2', predict_data)
    print(time.time() - start_time)
