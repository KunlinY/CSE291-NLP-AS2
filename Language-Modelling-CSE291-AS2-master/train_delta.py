import os
import json
import time
import torch
import argparse
import logging
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from ptb import PTB
from utils import to_var, idx2word, experiment_name
from model import SentenceVAE

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    model = SentenceVAE(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, str(args.delta)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    total_step = int(args.epochs * 42000.0 / args.batch_size)

    def kl_anneal_function(anneal_function, step):
        if anneal_function == 'half':
            return 0.5
        if anneal_function == 'identity':
            return 1
        if anneal_function == 'double':
            return 2
        if anneal_function == 'quadra':
            return 4

        if anneal_function == 'sigmoid':
            return 1/(1 + np.exp(0.5 * total_step - step))

        if anneal_function == 'monotonic':
            beta = step * 8 / total_step
            if step > 1:
                beta = 1.0
            return beta

        if anneal_function == 'cyclical':
            t = total_step / 4
            beta = (step % t) / t * 16 / total_step
            if step > 1:
                beta = 1.0
            return beta

    ReconLoss = torch.nn.NLLLoss(reduction='sum', ignore_index=datasets['train'].pad_idx)
    def loss_fn(logp, target, length, mean, logv, anneal_function, step):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        
        # Negative Log Likelihood
        recon_loss = ReconLoss(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step)

        return recon_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    train_loss = []
    test_loss = []
    for epoch in range(args.epochs):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(list)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['input'], batch['length'])

                # loss calculation
                recon_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step)

                alpha = args.delta
                KL_loss = max(KL_loss.item(), 0.5 * args.hidden_size * ((batch['length'] - 2) * np.log(1 + alpha * alpha) - np.log(1 - alpha * alpha)))

                if split == 'train':
                    loss = (recon_loss + KL_weight * KL_loss)/batch_size
                else:
                    # report complete elbo when validation
                    loss = (recon_loss + KL_loss)/batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1


                # bookkeepeing
                # tracker['negELBO'] = torch.cat((tracker['negELBO'], loss.data))
                tracker["negELBO"].append(loss.item())
                tracker["recon_loss"].append(recon_loss.item()/batch_size)
                tracker["KL_Loss"].append(KL_loss.item()/batch_size)
                tracker["KL_Weight"].append(KL_weight)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/Negative_ELBO"%split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Recon_Loss"%split.upper(), recon_loss.item()/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL_Loss"%split.upper(), KL_loss.item()/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL_Weight"%split.upper(), KL_weight, epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    logger.info("%s Batch %04d/%i, Loss %9.4f, Recon-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                        %(split.upper(), iteration, len(data_loader)-1, loss.item(), recon_loss.item()/batch_size, KL_loss.item()/batch_size, KL_weight))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
                    tracker['z'].append(z.data.tolist())

            logger.info("%s Epoch %02d/%i, Mean Negative ELBO %9.4f"%(split.upper(), epoch, args.epochs, sum(tracker['negELBO']) / len(tracker['negELBO'])))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/NegELBO"%split.upper(), 1.0 *sum(tracker['negELBO']) / len(tracker['negELBO']), epoch)
                writer.add_scalar("%s-Epoch/recon_loss"%split.upper(), 1.0 *sum(tracker['recon_loss']) / len(tracker['recon_loss']), epoch)
                writer.add_scalar("%s-Epoch/KL_Loss"%split.upper(), 1.0 *sum(tracker['KL_Loss']) / len(tracker['KL_Loss']), epoch)
                writer.add_scalar("%s-Epoch/KL_Weight"%split.upper(), 1.0 *sum(tracker['KL_Weight']) / len(tracker['KL_Weight']), epoch)

            if split == 'train':
                train_loss.append(1.0 * sum(tracker['negELBO']) / len(tracker['negELBO']))
            else:
                test_loss.append(1.0 * sum(tracker['negELBO']) / len(tracker['negELBO']))
            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents':tracker['target_sents'], 'z':tracker['z']}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json'%epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
                torch.save(model.state_dict(), checkpoint_path)
                logger.info("Model saved at %s"%checkpoint_path)

    sns.set(style="whitegrid")
    df = pd.DataFrame()
    df["train"] = train_loss
    df["test"] = test_loss
    ax = sns.lineplot(data=df, legend=False)
    ax.set(xlabel='Epoch', ylabel='Loss')
    plt.legend(title='Split', loc='upper right', labels=['Train', 'Test'])
    plt.savefig(os.path.join(args.logdir, str(args.delta), "loss.png"), transparent=True, dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='identity')

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-log','--logdir', type=str, default='logs')
    parser.add_argument('-bin','--save_model_path', type=str, default='bin')
    parser.add_argument('-delta','--delta', type=float, default=0.2)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)