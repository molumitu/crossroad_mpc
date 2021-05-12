from multiprocessing import Pool
from tqdm import trange
def f(x, y):
    return x*y

if __name__ == '__main__':
    test_samples = 26

    with Pool(5) as p:
        print(p.starmap(f, [(1, 2), (3, 4)]))


# import numpy as np 
# n1 = np.array([1, 2,3 , 4])
# n1 = np.maximum(n1, 2)
# print(n1)

    




