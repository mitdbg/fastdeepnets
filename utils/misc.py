from utils.wrapping import wrap
import random

def tn(x):
    return x.cpu().numpy()[0]

class PreloadedDataloader():

    def __init__(self, dl, shuffle=True, do_wrap=True):
        self.data = list(dl)
        if do_wrap:
            self.data = [tuple(wrap(z) for z in x) for x in self.data]
        self.shuffle = shuffle
        self.dataset = dl.dataset

    def __iter__(self):
        copy = self.data[:]
        if self.shuffle:
            random.shuffle(copy)
        return iter(copy)
