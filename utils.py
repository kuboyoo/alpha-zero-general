import pickle 

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.3f}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def savePkl(path: str, obj: object):
    # pickle化してファイルに書き込み
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def loadPkl(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
