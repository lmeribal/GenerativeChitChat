import telebot
import os
from inference import beam_search_generate
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

token = ''

decoder = Decoder(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=pad_index,
)
encoder = Encoder(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=pad_index,
)
model = LanguageModel(
    encoder=encoder,
    decoder=decoder,
    vocab_size=vocab_size,
    embedding_dim=embedding_dim
)

# trained_model_path = "trained_model/best_language_model_state_dict.pth"
trained_model_path = "trained_model/last_language_model_state_dict.pth"
if os.path.exists("/workdir/"):
    trained_model_path = "/workdir/" + trained_model_path
model.load_state_dict(torch.load(trained_model_path, map_location=device))
model.eval()
model.to(device)

bot=telebot.TeleBot(token)

@bot.message_handler(content_types='text')
def start_message(message):
    response = beam_search_generate(message.text, model, tokenizer)
    bot.send_message(message.chat.id, response)

bot.infinity_polling()
