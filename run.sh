dataname=SCoPE2
feature=adj
epochs=50
loss=bce
bs=96
subdir="hgnn/${dataname}-${feature}-${loss}"

mkdir -p ./embeddings/${subdir}

echo Begin to train ${subdir} for ${epochs} epochs

conda activate py39

python ./hgnn/main_torch_zinb.py --data ${dataname} --feature ${feature} --iter ${epochs} --batch_size=${bs} --loss ${loss}

echo ${subdir} ends

mv *.npy ./embeddings/${subdir}

echo Move embeddings to dir ${subdir}.
