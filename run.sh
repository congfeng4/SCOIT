dataname=sc_GEM
feature=walk
epochs=100
loss=bce
bs=96
subdir="hgnn/${dataname}"

mkdir -p ./embeddings/${subdir}

echo Begin to train ${subdir} for ${epochs} epochs

conda activate py39

python ./hgnn/main_torch.py --data ${dataname} --feature ${feature} --iter ${epochs} --batch_size=${bs}

echo ${dataname} ends

mv *.npy ./embeddings/${subdir}

echo Move embeddings to dir.
