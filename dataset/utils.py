from matplotlib import pyplot as plt
import numpy as np


def show_dataset(ds, extract_sample=lambda x: x, samples=4):
    plt.figure(figsize=(10, 6))
    p = np.random.permutation(range(len(ds)))
    for i in range(samples):
        sample = extract_sample(ds[p[i]])
        ax = plt.subplot(1, samples, i + 1)
        ax.set_title('Sample #{}'.format(i))
        plt.imshow(sample)
    plt.show()
