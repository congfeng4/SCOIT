import numpy as np
import pandas as pd
from scoit import sc_multi_omics
from scoit.cell_analysis import save_scoit_embeddings
import time


def load_data():
    expression_data = np.array(pd.read_csv("data/sc_GEM/expression_data.csv", index_col=0))
    methylation_data = np.array(pd.read_csv("data/sc_GEM/methylation_data.csv", index_col=0))
    cell_stage = np.array(pd.read_csv("data/sc_GEM/cell_stage.csv", header=None))[0]
    # labels = []
    # for each in cell_stage:
    #     if each == "BJ":
    #         labels.append(0)
    #     if each == "d8":
    #         labels.append(1)
    #     if each == "d16T-" or each == "d16T+":
    #         labels.append(2)
    #     if each == "d24T-" or each == "d24T+":
    #         labels.append(3)
    #     if each == "IPS":
    #         labels.append(4)
    #     if each == "ES":
    #         labels.append(5)

    return expression_data, methylation_data, cell_stage #labels



if __name__ == "__main__":

    start_time = time.time()
    expression_data, methylation_data, labels = load_data()
    data = np.array([expression_data, methylation_data])
    print(data.shape)

    sc_model = sc_multi_omics()
    # predict_data = sc_model.fit(data, dist="negative_bionomial", n_epochs=1000, device="cpu")
    predict_data = sc_model.fit_complete(
        data,
        dist="negative_bionomial",
        lr=1e-1,
        n_epochs=3000,  # original settting is 1000 but not converge.
        lambda_C_regularizer=0.01,
        lambda_G_regularizer=0.01,
        lambda_O_regularizer=[0.01, 0.01],
        lambda_OC_regularizer=[1, 1],
        lambda_OG_regularizer=[1, 1],
    )

    save_scoit_embeddings(sc_model, "sc_GEM")
    print('time', time.time() - start_time)
