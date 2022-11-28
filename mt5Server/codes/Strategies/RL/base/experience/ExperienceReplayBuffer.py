import numpy as np
from mt5Server.codes.Strategies.RL.base.experience.ExperienceSource import ExperienceSourceCMK


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSourceCMK, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def buffer_sample(self, monitor_size):
        if len(self.buffer) <= monitor_size:
            return self.buffer
        keys = np.random.choice(len(self.buffer), monitor_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, entry):
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)


# class BufferAttn:
#     def __new__(cls, seqLen, featureSize):
#         cls.seqLen = seqLen
#         cls.featureSize = featureSize
#         cls.buffer = {}
#         cls.buffer['encoderInput'] = []
#         cls.buffer['status'] = []
#         return cls.buffer
#
#     def __len__(self):
#         return len(self.buffer['status'])
#
#     def append(self, entry):
#         self.buffer['encoderInput'] = np.empty(shape=(1, 60, 55))
#         self.buffer['encoderInput'] = np.empty(shape=(1, 2))


# class ExperienceReplayBufferAttn(ExperienceReplayBuffer):
#     def __init__(self, seqLen, featureSize, experience_source, buffer_size):
#         super(ExperienceReplayBufferAttn, self).__init__(experience_source, buffer_size)
#         self.buffer = {}
#
#     def _add(self, entry):
#         """
#         :param entry: { encoderInput: array(N, 60, 55),
#                         status: array(N, 2)
#                         }
#         :return:
#         """
#         if len(self.buffer) < self.capacity:
#             self.buffer.append()
