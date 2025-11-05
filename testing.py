import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 5]])

arr_pad = np.pad(arr, pad_width=((1, 2), (3, 2)), mode="constant")
print(arr_pad)


uni_dist = np.random.uniform(0, 1, size=(4, 3, 3, 3))
print(uni_dist[1][0] == uni_dist[1][2])