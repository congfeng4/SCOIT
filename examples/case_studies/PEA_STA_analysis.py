import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time


def load_data():
    expression_data = pd.read_csv("data/PEA_STA/expression_data.csv", index_col=0)
    protein_data = pd.read_csv("data/PEA_STA/protein_data.csv", index_col=0)
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

    return expression_data, protein_data, labels


if __name__ == "__main__":
    start_time = time.time()
    expression_data, protein_data, labels = load_data()
    data = np.array([expression_data, protein_data])

    sc_model = sc_multi_omics()
    # predict_data = sc_model.fit(data, dist="gaussian", n_epochs=1000)
    predict_data = sc_model.fit_complete(
        data,
        dist="gaussian",
        lr=1e-2,
        n_epochs=1000,
        lambda_C_regularizer=0.01,
        lambda_G_regularizer=1,
        lambda_O_regularizer=[1, 1],
        lambda_OC_regularizer=[1, 1],
        lambda_OG_regularizer=[0.01, 0.01],
        normalization=True
    )

    np.savetxt("cell_embeddings.csv", sc_model.C, delimiter=',')
    np.savetxt("local_gene_embeddings.csv", sc_model.OG, delimiter=',')
    np.savetxt("predict_data_expression.csv", predict_data[0])
    np.savetxt("predict_data_protein.csv", predict_data[1])
    print(time.time() - start_time)
