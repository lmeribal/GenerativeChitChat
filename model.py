import random
import numpy as np
import torch.nn.functional as F

import torch

from settings import device, max_len, batch_size

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx,
        )

        self.gru = torch.nn.GRU(self.embedding_dim, self.embedding_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, memory):
        x = x.unsqueeze(dim=0)
        x = self.embedding_layer(x)

        # print(x.shape, memory.shape)
        # train: torch.Size([1, 4, 512]) torch.Size([1, 4, 512])
        encoder_output, memory = self.gru(x, memory)

        return encoder_output, memory

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.embedding_dim, device=device)

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder_embedding_layer = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )

        self.gru = torch.nn.GRU(self.embedding_dim, self.embedding_dim)

        self.language_model_head = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=vocab_size, bias=False
        )
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.dropout = torch.nn.Dropout(0.5)
        self.attn = torch.nn.Linear(self.embedding_dim * 2, max_len)
        self.attn_combine = torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim)

    def forward(self, input, memory, encoder_outputs):
        embedding = self.dropout(self.decoder_embedding_layer(input.view(1, 1, -1)))

        attn_weights = F.softmax(
            self.attn(torch.cat((embedding[0][0], memory[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0).movedim(0, 1),
                                 encoder_outputs.transpose(0, 1))
        try:
            output = torch.cat((embedding[0][0], attn_applied.squeeze()), 1)
        except RuntimeError:
            output = torch.cat((embedding[0][0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)

        output, memory = self.gru(output, memory)

        pred = self.language_model_head(output[0])
        return  self.softmax(pred), memory, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.embedding_dim, device=device)

class LanguageModel(torch.nn.Module):
    def __init__(self, encoder, decoder, vocab_size, embedding_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

    def forward(self, x, y, teacher_forcing_ratio):
        batch_size = y.shape[0]
        trg_len = y.shape[1]
        src_len = x.shape[1]
        trg_vocab_size = self.vocab_size

        encoder_outputs = torch.zeros(max_len, batch_size, self.embedding_dim, device=device)
        # print(encoder_outputs.shape)
        memory = self.encoder.init_hidden(batch_size)

        # torch.Size([32, 4, 512]) torch.Size([1, 4, 512])
        # print(encoder_outputs.shape, memory.shape)

        # x = x.squeeze()

        for ei in range(src_len):
            # print(x.shape, x.T[ei].shape)
            # print('ei', x.T.shape, x.T[ei].shape)
            encoder_output, memory = self.encoder(x.T[ei], memory)
            # print(encoder_output.shape)
            # print(encoder_output[0].shape)
            # encoder_outputs[ei] = encoder_output[0, 0]
            encoder_outputs[ei] = encoder_output[0]
            # raise Exception()

        outputs = torch.zeros(trg_len , batch_size, trg_vocab_size).to(device)
        # print(outputs.shape)
        # outputs[0] = F.log_softmax(torch.tensor([[0, 0] + [1] + [0] * (trg_vocab_size - 3)]).float())
        # print(outputs[0].shape)
        inp = y[:, 0]
        # print('inp', inp.shape)

        for t in range(1, trg_len - 1):

            # torch.Size([4]) torch.Size([1, 4, 512]) torch.Size([32, 4, 512])
            # print(inp.shape, memory.shape, encoder_outputs.shape)

            output, memory, attn_weights = self.decoder(inp, memory, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            # print(top1)
            if top1.cpu().numpy()[0] == 3:
                break
            inp = y[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs
