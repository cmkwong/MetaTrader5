import copy
import numpy as np
import torch
from torch import nn
from mt5Server.codes.Strategies.RL.base.agents.BaseAgent import BaseAgent
from mt5Server.codes.Strategies.RL.base.agents.preprocessor import default_states_preprocessor, attention_states_preprocessor

class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, preprocessor=default_states_preprocessor):
        self.device = torch.device("cuda")
        self.model = dqn_model
        self.target_model = copy.deepcopy(dqn_model)
        self.action_selector = action_selector
        self.preprocessor = preprocessor

    def unpack_batch(self, batch):
        states, actions, rewards, dones, last_states = [], [], [], [], []
        for exp in batch:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(exp.state)  # the result will be masked anyway
            else:
                last_states.append(exp.last_state)
        return states, np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), last_states

    def calc_loss(self, batch, gamma):
        """
        :param batch: [state]
        :param gamma: float
        :return:
        """
        states, actions, rewards, dones, next_states = self.unpack_batch(batch)

        states_v = states
        next_states_v = next_states
        actions_v = torch.from_numpy(actions).to(self.device)
        rewards_v = torch.from_numpy(rewards).to(self.device)
        done_mask = torch.cuda.BoolTensor(dones)

        state_action_values = self.get_Q_value(states_v).gather(1, actions_v).squeeze(-1)
        next_state_actions = self.get_Q_value(next_states_v).max(1)[1]
        next_state_values = self.get_Q_value(next_states_v, tgt=True).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
        loss = nn.L1Loss()(state_action_values, expected_state_action_values)
        # if torch.isnan(loss):
        #     print('error')
        return nn.L1Loss()(state_action_values, expected_state_action_values)

    def getActionIndex(self, states, agent_states=None):
        """
        :param states: [torch tensor] with shape (1, 57)
        :param agent_states:
        :return:
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        states = self.preprocessor(states) # states is a list
        q_v = self.model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states

    def get_Q_value(self, states, tgt=False):
        """
        :param states: [state]
        :return: pyTorch tensor
        """
        states = self.preprocessor(states)  # states is a list
        if not tgt:
            q_v = self.model(states)
        else:
            q_v = self.target_model(states)
        return q_v

    def sync(self):
        """
        sync the model and target model
        """
        self.target_model.load_state_dict(self.model.state_dict())

class Supervised_DQNAgent(BaseAgent):
    def __init__(self, dqn_model, action_selector, sample_sheet, assistance_ratio=0.2):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.sample_sheet = sample_sheet # name tuple
        self.assistance_ratio = assistance_ratio

    def __call__(self, states, agent_states=None):
        batch_size = len(states)
        if agent_states is None:
            agent_states = [None] * batch_size
        sample_mask = np.random.random(batch_size) <= self.assistance_ratio
        sample_actions_ = []
        dates = [state.date for state in states[sample_mask]]
        for date in dates:
            for i, d in enumerate(self.sample_sheet.date):
                if d == date:
                    sample_actions_.append(self.sample_sheet.action[i])
        sample_actions = np.array(sample_actions_)   # convert into array

        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        actions[sample_mask] = sample_actions
        return actions, agent_states

class DQNAgentAttn(DQNAgent):
    def __init__(self, dqn_model, action_selector, preprocessor=attention_states_preprocessor):
        super(DQNAgentAttn, self).__init__(dqn_model, action_selector, preprocessor)

    # def unpack_batch(self, batch):
    #     pass
    #
    # def calc_loss(self, batch, gamma):
    #     pass

# class TargetNet:
#     """
#     Wrapper around model which provides copy of it instead of trained weights
#     """
#     def __init__(self, model):
#         self.model = model
#         self.target_model = copy.deepcopy(model)
#
#     def sync(self):
#         self.target_model.load_state_dict(self.model.state_dict())
#
#     def alpha_sync(self, alpha):
#         """
#         Blend params of target net with params from the model
#         :param alpha:
#         """
#         assert isinstance(alpha, float)
#         assert 0.0 < alpha <= 1.0
#         state = self.model.state_dict()
#         tgt_state = self.target_model.state_dict()
#         for k, v in state.items():
#             tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
#         self.target_model.load_state_dict(tgt_state)
