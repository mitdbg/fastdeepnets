from utils.wrapping import wrap
import random

def tn(x):
    return x.cpu().numpy()[0]

class PreloadedDataloader():

    def __init__(self, dl, shuffle=True, do_wrap=True, dataset=None):
        self.data = dl
        if not isinstance(self.data, list):
            self.data = list(self.data)
            if do_wrap:
                self.data = [tuple(wrap(z) for z in x) for x in self.data]
        self.shuffle = shuffle
        self.dataset = dl.dataset if dataset is None else dataset

    def __iter__(self):
        copy = self.data[:]
        if self.shuffle:
            random.shuffle(copy)
        return iter(copy)

    def split(self, portion):
        split = int(portion * len(self.data))
        return [PreloadedDataloader(x, self.shuffle, dataset=self.dataset) for x in [self.data[:split], self.data[split:]]]
