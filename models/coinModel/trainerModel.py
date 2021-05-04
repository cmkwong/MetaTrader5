import torch
from production.codes.models.coinModel import criterionModel

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def _model_mode(self, train_mode):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

    # def _dataset(self, train_mode):
    #     if train_mode:
    #         inputs = self.train_batch.input
    #         targets = self.train_batch.target
    #     else:
    #         inputs = self.test_batch.input
    #         targets = self.test_batch.target
    #     return inputs, targets

    def _learn(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def run(self, inputs, targets, batch_size, train_mode=True):
        total_loss = 0
        steps = 0
        for input, target in zip(inputs, targets):
            self._model_mode(train_mode)
            input, target = torch.from_numpy(input).requires_grad_().double(), torch.from_numpy(target).requires_grad_().double()
            hiddens = self.model.init_hiddens(batch_size)
            output = self.model(input, hiddens)
            loss = criterionModel.get_mse_loss(output, target.to(torch.device("cuda:0")))
            if train_mode: self._learn(loss)
            total_loss += loss.detach().cpu().item()
            steps += 1
        mean_loss = total_loss / steps
        return mean_loss