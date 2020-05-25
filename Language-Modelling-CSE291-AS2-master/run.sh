git pull
apt-get update
pip install -r requirements.txt
python3 train_rnn.py --tensorboard_logging --data_dir ../data --logdir rnn_log --epochs 50 --batch_size 512
python3 train.py --tensorboard_logging --data_dir ../data --logdir vae_log --epochs 50 --batch_size 512 --anneal_function half
python3 train.py --tensorboard_logging --data_dir ../data --logdir vae_log --epochs 50 --batch_size 512 --anneal_function identity
python3 train.py --tensorboard_logging --data_dir ../data --logdir vae_log --epochs 50 --batch_size 512 --anneal_function double
python3 train.py --tensorboard_logging --data_dir ../data --logdir vae_log --epochs 50 --batch_size 512 --anneal_function quadra
python3 train.py --tensorboard_logging --data_dir ../data --logdir vae_log --epochs 50 --batch_size 512 --anneal_function sigmoid
python3 train.py --tensorboard_logging --data_dir ../data --logdir vae_log --epochs 50 --batch_size 512 --anneal_function monotonic
python3 train.py --tensorboard_logging --data_dir ../data --logdir vae_log --epochs 50 --batch_size 512 --anneal_function cyclical
