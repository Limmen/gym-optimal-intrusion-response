import numpy as np
from scipy.stats import multivariate_normal

def test():
    x = np.linspace(0, 5, 10, endpoint=False)
    y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
    #print(y)
    z = multivariate_normal.cdf([[180, 80], [100, 50], [190, 90], [300, 90], [300, 190]], mean=[200, 100], cov=[[80, 0], [0, 50]])
    print(z.tolist())

if __name__ == '__main__':
    test()