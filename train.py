import numpy as np
import torch
from tqdm import tqdm

from settings import device, tokenizer

class Trainer:
    def __init__(self, model, train_loader, validation_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self, clip=3, last_n_losses=500, verbose=True):
        losses = []
        progress_bar = tqdm(
            total=len(self.train_loader), disable=not verbose, desc="Train"
        )

        self.model.train()

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(device)
            y = y.to(device)

            self.optimizer.zero_grad()

            pred = self.model(x, y, 0.8)

            if i % 500 == 0 and i != 0:
                check = pred[:, 0].cpu().detach().numpy()
                print(tokenizer.decode([[el.argmax() for el in check]]))

            y = y.T[1:]

            pred = pred[1:].view(-1, pred.size(-1))
            y = y.T.contiguous().view(-1)

            loss = self.criterion(pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()

            losses.append(loss.item())

            if i % 1000 == 0 and i != 0:
                print(i, len(self.train_loader), sum(np.array(losses)) / i)

            progress_bar.set_postfix(
                loss=np.mean(losses[-last_n_losses:]),
                perplexity=np.exp(np.mean(losses[-last_n_losses:])),
            )

            progress_bar.update()
        progress_bar.close()
        return losses

    def evaluate(self, last_n_losses=500, verbose=True):
        losses = []
        progress_bar = tqdm(
            total=len(self.validation_loader), disable=not verbose, desc="Evaluate"
        )
        self.model.eval()

        for x, y in self.validation_loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                # print(x.shape, y.shape)
                pred = self.model(x, y, 0)

            # pred = pred[1:]
            y = y.T[1:]
            # print(pred.shape, y.shape)

            pred = pred[1:].view(-1, pred.size(-1))
            y = y.T.contiguous().view(-1)
            # print(pred.shape, y.shape)

            loss = self.criterion(pred, y)
            losses.append(loss.item())
            progress_bar.set_postfix(
                loss=np.mean(losses[-last_n_losses:]),
                perplexity=np.exp(np.mean(losses[-last_n_losses:])),
            )
            progress_bar.update()
        progress_bar.close()
        return losses
