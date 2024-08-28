mkdir integrals

conda activate qode

python be2_integrals.py <distance>

./run <distance> <n1-n2-n3> <compression>

cd integrals
cp configs_1e.npy configs_1e-stable.npy
cp configs_2e.npy configs_2e-stable.npy
cp configs_3e.npy configs_3e-stable.npy
cd ..

python Be631g.py <compression>
