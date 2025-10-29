dataname=sc_GEM
feature=walk
epochs=200
loss=mse
bs=96
subdir="hgnn/${dataname}-${loss}"

mkdir -p ./embeddings/${subdir}

echo Begin to train ${subdir} for ${epochs} epochs

conda activate py39

python ./hgnn/main_torch_zinb.py --data ${dataname} --feature ${feature} --iter ${epochs} --batch_size=${bs} --loss ${loss}

echo ${dataname} ends

mv *.npy ./embeddings/${subdir}

echo Move embeddings to dir.
