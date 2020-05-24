git pull
apt-get update
pip install -r requirements.txt
python3 train_rnn.py --tensorboard_logging --data_dir ../data --logdir rnn_log
python3 train.py --tensorboard_logging --data_dir ../data --logdir rnn_log --epochs 50
