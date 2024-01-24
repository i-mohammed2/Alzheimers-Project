import pickle

with open('calibration.pk1', 'rb') as f:
    data = pickle.load(f)
    print(data)