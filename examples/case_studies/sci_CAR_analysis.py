import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time

from scoit.cell_analysis import save_scoit_embeddings


def load_data():
    expression_data = pd.read_csv("data/sci_CAR/RNA_pca.csv", index_col=0, na_filter=False).to_numpy()

    ATAC_data = pd.read_csv("data/sci_CAR/ATAC_lsi.csv", index_col=0, na_filter=False).to_numpy()[:, 1:]

    labels = np.loadtxt("data/sci_CAR/label.txt")

    return expression_data, ATAC_data, labels


if __name__ == "__main__":
    start_time = time.time()
    expression_data, ATAC_data, labels = load_data()
    data = [expression_data, ATAC_data]
    print(data[0].shape)
    print(data[1].shape)

    sc_model = sc_multi_omics()
    predict_data = sc_model.fit_list_complete(
        data,
        normalization=False,
        dist="gaussian",
        lr=1e-3,
        n_epochs=3000,
        lambda_C_regularizer=0.01,
        lambda_G_regularizer=0.01,
        lambda_O_regularizer=[0.01, 0.01],
        pre_impute=False,
    )
    print(sc_model.G[0].shape, sc_model.G[1].shape)

    save_scoit_embeddings(sc_model, 'sci_CAR', predict_data)
    print('time', time.time() - start_time)
