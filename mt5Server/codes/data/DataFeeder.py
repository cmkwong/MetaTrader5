from myUtils.DfModel import DfModel


class DataFeeder(DfModel):

    def __init__(self, df):
        self.df = df
        self.i = 0

    def toTimeSeries(self, batchSize: int, t: int):
        return self.df.iloc[self.i - t:self.i, :]
