dataname=sc_GEM
feature=adj
epochs=30
subdir="${dataname}-${feature}"

mkdir -p ./logs/${subdir}/ ./embeddings/${subdir}

echo Begin to train ${dataname} for ${epochs} epochs
cd hgnn

python main_torch.py --data ${dataname} --feature ${feature} --iter ${epochs}

echo ${dataname} ends

cd ..
mv *.npy ./embeddings/${subdir}

echo Move embeddings to dir.
