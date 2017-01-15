import pickle

def load_model(path = './trained_rbm'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
    
def normalise_data(arr):
    min_ = arr.min()
    max_ = arr.max()
    if not (min_ >= 0 and max_ <= 1):
        return (arr - min_) / (max_ - min_)
    else:
        return arr
