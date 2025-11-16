from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim import Adam
from torch.nn.functional import mse_loss, relu
import torch.nn.functional as F
import torch

from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
import numpy as np


# 1. 由细胞嵌入 C 构建 kNN 图拉普拉斯（最简单）
def build_laplacian(C_np, k=10):
    """
    C_np: (n_cells, embed_dim) 的 numpy 数组
    返回: (n_cells, n_cells) 的稀疏 FloatTensor
    """
    adj = kneighbors_graph(C_np, k, include_self=False).toarray()
    adj = np.maximum(adj, adj.T)   # 强制对称
    lap = csgraph.laplacian(adj, normed=True)      # 归一化拉普拉斯
    return torch.tensor(lap, dtype=torch.float)


class PhasingModel(nn.Module):

    def __init__(self, embed_dim, activations, omics_offset):
        super().__init__()
        self.omics_offset = omics_offset
        self.activations = activations

        self.M = {} # d x d matrix
        self.Lambda = {} # scalar

        for parental in range(2):
            for omic in range(2):
                self.M[parental, omic] = nn.Parameter(torch.empty(embed_dim, embed_dim))
                self.Lambda[parental, omic] = nn.Parameter(torch.empty(1))

        self.A = {0: nn.Parameter(torch.empty(1)),
                  1: nn.Parameter(torch.empty(1))}
        self.B = {0: nn.Parameter(torch.empty(1)),
                  1: nn.Parameter(torch.empty(1))}

        for param in self.parameters():
            nn.init.uniform_(param)

    def parameters(self, recurse: bool = True):
        yield from self.M.values()
        yield from self.A.values()
        yield from self.B.values()
        yield from self.Lambda.values()

    def forward(self, G, C, O):
        act0, act1 = self.activations
        G0, G1 = G[:self.omics_offset], G[self.omics_offset:]
        Xp0 = act0(C @ self.M[0, 0] @ (G0 + self.Lambda[0, 0] * O[0]).T * self.A[0] + self.B[0])
        Xp1 = act1(C @ self.M[0, 1] @ (G1 + self.Lambda[0, 1] * O[1]).T * self.A[1] + self.B[1])
        Xm0 = act0(C @ self.M[1, 0] @ (G0 + self.Lambda[1, 0] * O[0]).T * self.A[0] + self.B[0])
        Xm1 = act1(C @ self.M[1, 1] @ (G1 + self.Lambda[1, 1] * O[1]).T * self.A[1] + self.B[1])
        return Xp0, Xp1, Xm0, Xm1

    @torch.no_grad()
    def predict(self, G, C, O):
        res = self.forward(G, C, O)
        return [r.cpu().numpy() for r in res]

    def reconstruct(self, G, C, O):
        return self.forward(G, C, O)


def train_phasing_model(*,embed_dim: int, omics_offset: int,
                        epochs: int, lr: float, k: int, alpha: float,
                        C, G, O, X,
                        recon_loss=None,
                        activations=None):
    if activations is None:
        activations = [F.softplus, F.sigmoid]
    if recon_loss is None:
        recon_loss = [F.mse_loss, F.mse_loss]

    model = PhasingModel(embed_dim=embed_dim, activations=activations, omics_offset=omics_offset)
    optimizer = Adam(model.parameters(), lr=lr)
    num_cells = X.shape[0]
    cell_dataset = TensorDataset(C, X)
    dataloader = DataLoader(cell_dataset, shuffle=True, batch_size=num_cells)

    history, history_recon, history_reg = [], [], []

    for epoch in range(epochs):
        epoch_loss, epoch_rec, epoch_reg = 0., 0., 0.
        for c_batch, x_batch in dataloader:
            c_batch, x_batch = c_batch.to(C.device), x_batch.to(C.device)

            xp0, xp1, xm0, xm1 = model(G, c_batch, O)
            x0_hat, x1_hat = xp0 + xm0, xp1 + xm1
            xp = torch.cat([xp0, xp1], dim=1)
            xm = torch.cat([xm0, xm1], dim=1)
            x0_batch, x1_batch = x_batch[:, :model.omics_offset], x_batch[:, model.omics_offset:]
            loss_recon = recon_loss[0](x0_hat, x0_batch) + recon_loss[1](x1_hat, x1_batch)

            lap0_batch = build_laplacian(x0_batch.cpu().numpy(), k=k).to(C.device)
            loss_regular0 = torch.trace(xp0.T @ lap0_batch @ xp0) + torch.trace(xm0.T @ lap0_batch @ xm0)
            lap1_batch = build_laplacian(x1_batch.cpu().numpy(), k=k).to(C.device)
            loss_regular1 = torch.trace(xp1.T @ lap1_batch @ xp1) + torch.trace(xm1.T @ lap1_batch @ xm1)
            loss_reg = loss_regular0 + loss_regular1

            loss = loss_recon + alpha * loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rec += loss_recon.item()
            epoch_reg += loss_reg.item()

        history.append(epoch_loss)
        history_recon.append(epoch_rec)
        history_reg.append(epoch_reg)
        if epoch % 100 == 0:
            print(f'Epoch {epoch:3d}  total={epoch_loss:.3f}  rec={epoch_rec:.3f}  reg={epoch_reg:.3f}')

    return model, dict(history=history, history_recon=history_recon, history_reg=history_reg)
