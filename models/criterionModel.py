from torch import nn

def get_mse_loss(predicts, targets):
    mse = nn.MSELoss()
    loss = mse(predicts, targets)
    return loss