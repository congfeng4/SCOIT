dataname=sc_GEM
feature=walk
epochs=100
loss=bce
subdir="hgnn/${dataname}-${loss}"

mkdir -p ./embeddings/${subdir}

echo Begin to train ${subdir} for ${epochs} epochs

python main_torch.py --data ${dataname} --feature ${feature} --iter ${epochs} --loss ${loss}

echo ${dataname} ends

mv *.npy ./embeddings/${subdir}

echo Move embeddings to dir.
