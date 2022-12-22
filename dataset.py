import torch
from settings import max_len


class LanguageModelData(torch.utils.data.Dataset):
    def __init__(self, data, max_len, pad_index, eos_index):
        self.data = data

        self.max_len = max_len

        self.pad_index = pad_index
        self.eos_index = eos_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data.iloc[index]

        x = list(sequence["question_tokenized"])[:self.max_len]
        y = list(sequence["answer_tokenized"])[:self.max_len]

        pads_x = [self.pad_index] * (self.max_len - len(x))
        pads_y = [self.pad_index] * (self.max_len - len(y))

        x = torch.tensor(x + pads_x).long()
        y = torch.tensor(y + pads_y).long()

        return x, y
