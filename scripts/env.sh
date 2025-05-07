sudo apt install python3.11-venv -y

python3 -m venv env

source env/bin/activate

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/compat

pip install -e ".[train]"

cd transformers

pip install -e .

cd ..
