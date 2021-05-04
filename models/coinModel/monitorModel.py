import torch
import numpy as np

def get_predicted_arr(input_matrix, model, seq_len):
    model.eval()
    predict_arr = np.zeros((len(input_matrix), 1), dtype=np.double)
    for i in range(seq_len, len(input_matrix)):
        input = input_matrix[i-seq_len:i,:]
        input = torch.from_numpy(input).unsqueeze(0).double() # size = (batch_size, seq_len, 1)
        hiddens = model.init_hiddens(1)
        predict = model(input, hiddens)
        predict_arr[i, 0] = predict
    return predict_arr

