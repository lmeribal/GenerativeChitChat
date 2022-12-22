import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm
import boto3
from dataset import LanguageModelData
from model import Decoder, Encoder, LanguageModel
from settings import (
    device,
    embedding_dim,
    eos_index,
    epochs,
    img_path,
    num_layers,
    pad_index,
    tokenizer,
    vocab_size,
    max_len,
    batch_size
)
from train import Trainer


data_path = "data/seq2seq_data.zip"

dropout = 0.5

logger.debug('S3 connection...')
# s3 = boto3.client('s3',
#                   aws_access_key_id='',
#                   aws_secret_access_key='',
#                   endpoint_url = '')
# bucket_name = ''
file_name = 'seq2seq_data.zip'

logger.debug('S3 data downloading...')
obj = s3.get_object(Bucket=bucket_name, Key=file_name)

if os.path.exists('/workdir'):
    s3.download_file(bucket_name, 'pretrained_lm.pth', '/workdir/pretrained_lm.pth')

logger.debug('Downloaded!')

def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)

def process_dataset(df):
    df = df[~df["answer"].isna()]
    df = df[~df["question"].isna()]

    logger.debug("Opening tokenizer...")

    question_tokenized = []
    answer_tokenized = []

    tokenizer_batch_size = 256

    for i_batch in tqdm(range(len(df["question"]) // tokenizer_batch_size)):
        question_tokenized.extend(
            tokenizer.encode(
                list(
                    df["question"].loc[
                        i_batch
                        * tokenizer_batch_size : (i_batch + 1)
                        * tokenizer_batch_size
                    ]
                ),
                bos=True,
                eos=True,
            )
        )

    for i_batch in tqdm(range(len(df["answer"]) // tokenizer_batch_size)):
        answer_tokenized.extend(
            tokenizer.encode(
                list(
                    df["answer"].loc[
                        i_batch
                        * tokenizer_batch_size : (i_batch + 1)
                        * tokenizer_batch_size
                    ]
                ),
                bos=True,
                eos=True,
            )
        )
    print(df[["question", "answer"]][:5])

    tokenized_df = pd.DataFrame()
    tokenized_df["question_tokenized"] = question_tokenized
    tokenized_df["answer_tokenized"] = answer_tokenized

    logger.debug(f"Tokenized length: {len(tokenized_df)}")

    validation_start_index = int(len(tokenized_df) * 0.1)
    logger.debug(f"Train length: {len(tokenized_df[:-validation_start_index])}")
    logger.debug(f"Val length: {len(tokenized_df[-validation_start_index:])}")

    train_dataset = LanguageModelData(
        data=tokenized_df[:-validation_start_index],
        max_len=max_len,
        pad_index=pad_index,
        eos_index=eos_index,
    )

    validation_dataset = LanguageModelData(
        data=tokenized_df[-validation_start_index:],
        max_len=max_len,
        pad_index=pad_index,
        eos_index=eos_index,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader, validation_loader


def write_loss_graph(losses, n_epoch, type):
    plt.figure(figsize=(14, 14))
    plt.xlabel("Номер батча")
    plt.ylabel("Значение функции потерь")
    plt.title("Процесс обучения")
    plt.plot(losses)
    plt.savefig(f"{img_path}/{type}_loss_epoch_{n_epoch}.png")


def main():
    logger.debug("Data reading...")

    data = []
    # for chunk in tqdm(pd.read_csv(obj['Body'], chunksize=100000, sep='\t', compression='gzip', nrows=200_000)):
    for chunk in tqdm(pd.read_csv(obj['Body'], chunksize=100000, sep='\t', compression='gzip')):
    # for chunk in tqdm(
    #     pd.read_csv(data_path, chunksize=100000, sep="\t", compression="gzip")
    # ):
        data.append(chunk)
    df = pd.concat(data)

    logger.debug("Tokenizing...")
    train_loader, validation_loader = process_dataset(df)

    logger.debug(f"Train batches: {len(train_loader)}")
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=pad_index
    )

    if os.path.exists("/workdir/"):
        pretrained_path = '/workdir/pretrained_lm.pth'
    else:
        pretrained_path = 'pretrained_lm.pth'

    encoder.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)

    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=pad_index
    )

    decoder.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)

    model = LanguageModel(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim
    ).to(device)

    model.apply(init_weights)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = torch.optim.Adam(params=model.parameters())

    trainer = Trainer(model, train_loader, validation_loader, criterion, optimizer)

    train_losses = []
    validation_losses = []

    train_perplexities = []
    validation_perplexities = []

    best_validation_loss = 1e6

    for n_epoch in range(1, epochs + 1):
        logger.debug("Train loop...")
        epoch_train_losses = trainer.train()
        logger.debug("Evaluate...")
        epoch_validation_losses = trainer.evaluate()

        write_loss_graph(epoch_train_losses, "train", n_epoch)
        write_loss_graph(epoch_validation_losses, "val", n_epoch)

        mean_train_loss = np.mean(epoch_train_losses)
        mean_validation_loss = np.mean(epoch_validation_losses)

        train_losses.append(epoch_train_losses)
        train_perplexities.append(np.exp(mean_train_loss))

        validation_losses.append(epoch_validation_losses)
        validation_perplexities.append(np.exp(mean_validation_loss))

        message = f"Epoch: {n_epoch}\n"
        message += f"Train: loss - {mean_train_loss:.4f} | perplexity - {train_perplexities[-1]:.3f}\n"
        message += f"Validation: loss - {mean_validation_loss:.4f} | perplexity - {validation_perplexities[-1]:.3f}"

        logger.debug(message)
        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            best_model_path = "trained_model/best_language_model_state_dict.pth"
            best_optimizer_path = "trained_model/best_optimizer_state_dict.pth"
            if os.path.exists("/workdir/"):
                best_model_path = "/workdir/" + best_model_path
                best_optimizer_path = "/workdir/" + best_optimizer_path
            torch.save(model.state_dict(), best_model_path)
            torch.save(optimizer.state_dict(), best_optimizer_path)

        model_path = f"trained_model/last_language_model_state_dict.pth"
        optimizer_path = "trained_model/last_optimizer_state_dict.pth"
        if os.path.exists("/workdir/"):
            model_path = "/workdir/" + model_path
            optimizer_path = "/workdir/" + optimizer_path

        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)

        log_path = f"trained_model/info_{n_epoch}.json"
        if os.path.exists("/workdir/"):
            log_path = "/workdir/" + log_path
        with open(log_path, "w") as file_object:
            info = {
                "message": message,
                "train_losses": train_losses,
                "validation_losses": validation_losses,
                "train_perplexities": train_perplexities,
                "validation_perplexities": validation_perplexities,
            }
            file_object.write(json.dumps(info, indent=2))
    shutil.make_archive("/workdir/images", "zip", img_path)


if __name__ == "__main__":
    main()
