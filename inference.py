import os

import numpy as np
import torch

from model import Decoder, Encoder, LanguageModel
from settings import (
    device,
    embedding_dim,
    model_dim,
    num_layers,
    pad_index,
    tokenizer,
    vocab_size,
)

softmax = torch.nn.LogSoftmax(dim=1)


def beam_search_generate(
    seed_text,
    model,
    tokenizer,
    bos_index=2,
    eos_index=3,
    max_sequence=32,
    beams=1,
    top_k=3,
    temperature=1,
):
    print(seed_text)
    tokenized = tokenizer.encode([seed_text])
    tokenized[0].insert(0, bos_index)
    tokenized[0].append(eos_index)
    x = torch.tensor([tokenized[0] + [0] * (max_sequence - len(tokenized[0]))]).long().to(device)
    # print(x.shape)
    beam_search_dict = {}

    with torch.no_grad():
        encoder_outputs = torch.zeros(x.shape[-1], 1, 512, device=device)
        memory = model.encoder.init_hidden(batch_size=1)

        # torch.Size([1, 512]) torch.Size([1, 1, 512])
        # print(encoder_outputs.shape, memory.shape)
        for ei in range(len(x)):
            encoder_output, memory = model.encoder(x.T[ei], memory)
            encoder_outputs[ei] = encoder_output[0]

        for k in range(beams):
            pred = []
            proba = 0
            current_token = x[0][0].unsqueeze(0)

            # torch.Size([32]) torch.Size([1, 1, 512]) torch.Size([1, 1, 512])
            # print(current_token.shape, memory.shape, encoder_outputs.shape)

            next_token_pred, memory, attn_weights = model.decoder(current_token, memory, encoder_outputs)
            token_proba = np.array(next_token_pred.squeeze(0).squeeze(0).cpu())

            current_token = token_proba.argsort()[-beams:][k]
            pred.append(current_token)
            for timestamp in range(max_sequence):

                # print(current_token.shape, memory.shape, encoder_outputs.shape)

                next_token_pred, memory, attn_weights = model.decoder(
                    torch.tensor([current_token]).to(device), memory, encoder_outputs
                )

                token_proba = np.array(next_token_pred.squeeze(0).squeeze(0).cpu())
                current_token = np.random.choice(token_proba.argsort()[-top_k:])

                if current_token == eos_index:
                    break

                proba += token_proba[current_token]
                pred.append(current_token)
            beam_search_dict[tokenizer.decode([pred])[0]] = proba
    return max(beam_search_dict, key=beam_search_dict.get)

def inference():
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=pad_index,
    )
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=pad_index
    )
    model = LanguageModel(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim
    )

    model.to(device)
    model.eval()
    # trained_model_path = "trained_model/best_language_model_state_dict.pth"
    trained_model_path = "trained_model/last_language_model_state_dict.pth"
    if os.path.exists("/workdir/"):
        trained_model_path = "/workdir/" + trained_model_path
    model.load_state_dict(torch.load(trained_model_path, map_location=device))

    print(beam_search_generate("Что делать если собака перестала лаять?", model, tokenizer))
    print(beam_search_generate("Где можно купить маленьких котят?", model, tokenizer))
    print(beam_search_generate("Какие прививки нужно ставить молодой кошке?", model, tokenizer))
    print(beam_search_generate("Как прогнать змей из подвала дачного дома?", model, tokenizer))
    print(beam_search_generate("Кто такие летучие мыши?", model, tokenizer))



if __name__ == "__main__":
    inference()
