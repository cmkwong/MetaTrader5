import numpy as np
import torch
from torch.nn import functional as func


def default_states_preprocessor(states, unitVector=True, train_on_gpu=True):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    assert isinstance(states, list)
    # choose device
    if train_on_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # pre-process the states
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    t_v = func.normalize(torch.from_numpy(np_states).to(device), p=2, dim=1) if unitVector else torch.from_numpy(np_states).to(device)
    return t_v

def attention_states_preprocessor(states):
    """
    :param states: [{ encoderInput: np.array(), status: np.array() }]
    :return:
    """
    device = torch.device('cuda')
    # pre-process the states
    if len(states) == 0:
        print("states cannot be empty")
        return False
    # assign the first states
    np_encoderInput = np.expand_dims(states[0]['encoderInput'], 0)
    np_status = np.expand_dims(states[0]['status'], 0)
    # if there is more than one, loop for concat
    for s in states[1:]:
        np_encoderInput = np.concatenate((np_encoderInput, np.expand_dims(s['encoderInput'], 0)), axis=0)
        np_status = np.concatenate((np_status, np.expand_dims(s['status'], 0)), axis=0)
    t_encoderInput = torch.from_numpy(np_encoderInput).type(torch.float32).to(device)
    t_status = torch.from_numpy(np_status).type(torch.float32).to(device)
    return {'encoderInput': t_encoderInput, 'status': t_status}

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.from_numpy(np_states)

