dataname=PEA_STA
feature=walk
epochs=30
subdir="${dataname}-${feature}"

mkdir -p ./logs/${subdir}/ ./embeddings/${subdir}

echo Begin to train ${subdir} for ${epochs} epochs
cd hgnn

python main_torch.py --data ${dataname} --feature ${feature} --iter ${epochs}

echo ${dataname} ends

cd ..
mv *.npy ./embeddings/${subdir}

echo Move embeddings to dir.
