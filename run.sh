dataname=PEA_STA
feature=walk
epochs=30
loss=zinb
subdir="${dataname}-${loss}"

mkdir -p ./logs/${subdir}/ ./embeddings/${subdir}

echo Begin to train ${subdir} for ${epochs} epochs
cd hgnn

python main_torch.py --data ${dataname} --feature ${feature} --iter ${epochs} --loss ${loss}

echo ${dataname} ends

cd ..
mv *.npy ./embeddings/${subdir}

echo Move embeddings to dir.
