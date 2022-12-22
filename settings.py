import os

import torch
import youtokentome as yttm

embedding_dim = 512
vocab_size = 30_000

pad_index = 0
max_len = 32
eos_index = 3

num_layers = 2
epochs = 10

model_dim = 512
batch_size = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_path = "images"

tokenizer_path = "pretrained_bpe_lm.model"
if os.path.exists("/workdir/"):
    img_path = "/workdir/" + img_path
    tokenizer_path = "/workdir/" + tokenizer_path

tokenizer = yttm.BPE(model=tokenizer_path)
