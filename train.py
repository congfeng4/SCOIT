import os
import subprocess
from pathlib import Path
import sys
import json


def train_embeddings(data: str, niter: int = 300,
                     feature: str = 'walk',
                     loss: str = 'bce',
                     bs: int = 96):
    assert Path('./data').joinpath(data).exists(), data

    subdir = f'{data}-{loss}'
    embed_dir = Path('./embeddings/hgnn').joinpath(subdir)
    embed_dir.mkdir(parents=True, exist_ok=True)
    print('Begin to train', subdir, 'for', niter, 'epochs')

    os.system('rm -f ./*.npy')

    subprocess.check_call([
        sys.executable, './hgnn/main_torch_zinb.py',
        '--data', data,
        '-f', feature,
        '--iter', str(niter),
        '--loss', loss,
        '--batch_size', str(bs),
    ])
    print('Train', data, 'ends')

    config = dict(
        data=data,
        niter=niter,
        feature=feature,
        loss=loss,
        bs=bs,
    )
    config_file = embed_dir/'config.json'
    config_file.write_text(json.dumps(config))

    os.system(f'mv *.npy {embed_dir}')
    print('Move embeddings to dir.')


ALL_DATA = 'SCoPE2 '.split()
# ALL_DATA = 'scNMT  SCoPE2  SNARE_seq_adult_mouse  SNARE_seq_neonatal_mouse CITE_seq'.split()
# PEA_STA  sc_GEM  sci_CAR
ALL_LOSS = 'bce mse'.split()


def all_combinations():
    for data in ALL_DATA:
        loss_list = ALL_LOSS.copy()
        if data in 'PEA_STA sc_GEM'.split():
            loss_list.append('zinb')
        else:
            loss_list.append('gauss')

        for loss in loss_list:
            yield dict(data=data, loss=loss)


if __name__ == '__main__':
    train_embeddings(
        data='PEA_STA', loss='bce',
    )
