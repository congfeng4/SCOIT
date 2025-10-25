dataname=PEA_STA
feature=walk
epochs=10
loss=bce
bs=96
subdir="hgnn/${dataname}-${loss}"

mkdir -p ./embeddings/${subdir}

echo Begin to train ${subdir} for ${epochs} epochs

python ./hgnn/main_torch.py --data ${dataname} --feature ${feature} --iter ${epochs} --loss ${loss} --batch_size=${bs}

echo ${dataname} ends

mv *.npy ./embeddings/${subdir}

echo Move embeddings to dir.
