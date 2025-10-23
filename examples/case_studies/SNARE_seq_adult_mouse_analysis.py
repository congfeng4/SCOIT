import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time

from scoit.cell_analysis import save_scoit_embeddings

def load_data():

    expression_data = pd.read_csv("data/SNARE_seq_adult_mouse/RNA_pca.csv", index_col=0, na_filter=False).to_numpy()

    ATAC_data = pd.read_csv("data/SNARE_seq_adult_mouse/ATAC_lsi.csv", index_col=0, na_filter=False).to_numpy()[:, 1:]

    labels = np.loadtxt("data/SNARE_seq_adult_mouse/label.txt")

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
        dist="gaussian",
        lr=1e-1,
        n_epochs=3000,
        lambda_C_regularizer=0.01,
        lambda_G_regularizer=0.01,
        lambda_O_regularizer=[0.01, 0.01],
        # pre_impute=True,
    )

    save_scoit_embeddings(sc_model, "SNARE_seq_adult_mouse_analysis", predict_data)
    print('time', time.time() - start_time)
