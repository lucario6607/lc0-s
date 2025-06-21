sudo apt-get update

sudo apt-get install -y python3-venv python3-pip git protobuf-compiler

git clone https://github.com/LeelaChessZero/lczero-training.git

cd lczero-training

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

conda create -n lczero python=3.10

conda activate lczero

pip install --upgrade pip

pip install tensorflow==2.11.* pyyaml "numpy<2"

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

tar -xvf /notebooks/training-run1-test80-20220404-0817.tar -C /notebooks/

protoc -I=. --python_out=. net.proto

cd /notebooks/lczero-training/tf

python train.py --cfg configs/example.yaml
