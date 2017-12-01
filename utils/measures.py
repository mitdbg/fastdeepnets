import pandas as pd
import numpy as np
from time import time

class TimeContextManager:
    def __init__(self, stats, metric, func):
        self.stats = stats
        self.func = func
        self.metric = metric

    def __enter__(self):
        self.start = time()

    def __exit__(self, _, __, ___):
        duration = time() - self.start
        self.stats.log("time_%s" % self.metric, duration)

    def __call__(self, *args, **kargs):
            return self.func(*args, **kargs)

class TrainingStats():

    def __init__(self):
        self.epoch = 0
        self.batch = 0
        self.events = []
        self.start = None

    def next_epoch(self):
        self.epoch += 1

    def next_batch(self):
        self.batch +=1

    def log(self, measure, value):
        t = time()
        if self.start is None:
            self.start = t
        self.events.append([t - self.start, self.batch, self.epoch, measure, value])

    def time(self, metric):
        def with_func(func):
            def job(*args, **kwargs):
                start = time()
                result = func(*args, **kwargs)
                duration = time() - start
                self.log('time_%s' % metric, duration)
                return result
            return job
        return TimeContextManager(self, metric, with_func)

    @property
    def logs(self):
        data = pd.DataFrame(np.array(self.events), columns=(
            'time', 'batch', 'epoch', 'measure', 'value'))
        data.batch = data.batch.astype(int)
        data.epoch = data.epoch.astype(int)
        data.value = data.value.astype(float)
        data.meausure = data.measure.astype(str)
        return data

