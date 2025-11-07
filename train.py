import os
from pprint import pp
import subprocess
from pathlib import Path
import sys
import json


def train_embeddings(data: str,
                     niter: int = 1,
                     feature: str = 'adj',
                     loss: str = 'bce',
                     bs: int = 1024):
    assert Path('./data').joinpath(data).exists(), data

    subdir = f'{data}-{feature}-{loss}'
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
    print('Train', subdir, 'ends')

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


ALL_DATA = 'sci_CAR scNMT  SCoPE2  SNARE_seq_adult_mouse  SNARE_seq_neonatal_mouse CITE_seq'.split()
# sc_GEM PEA_STA

ALL_LOSS = 'bce mse'.split()


def train_all(dry_run=False):
    if dry_run:
        niter = 1
        bs = 1024
    else:
        niter = 50
        bs = 256

    all_combs = list(all_combinations())
    pp(all_combs)

    for combo in all_combs:
        train_embeddings(niter=niter, bs=bs, **combo)


def all_combinations():
    zinb_loss_data = 'PEA_STA sc_GEM'.split()

    for data in ALL_DATA:
        loss_list = ALL_LOSS.copy()
        if data in zinb_loss_data:
            loss_list.append('zinb')
        else:
            loss_list.append('gauss')

        for loss in loss_list:
            yield dict(data=data, loss=loss)


if __name__ == '__main__':
    train_all(dry_run=True)
