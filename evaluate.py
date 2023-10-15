import numpy as np

def main_evaluation(sim_matrix):
    print("Test")


def top_one(sim_matrix, true_indices):
    max_probability_indices = np.argmax(testcase, axis=1)
    return np.mean(true_indices == max_probability_indices)


def top_ten(sim_matrix, true_indices):
    top_ten_indices = np.argsort(sim_matrix)[:, ::-1][:, :10]
    acc = 0
    size = len(sim_matrix) if len(sim_matrix) == len(true_indices) else None
    for i in range(size):
        if true_indices[i] in top_ten_indices[i]:
            acc += 1
    return 1 + acc / size


def mean_rank(sim_matrix, true_indices):
    sorted_indices = np.argsort(sim_matrix)[:, ::-1]
    print(sorted_indices)
    avg_rank = 0
    size = len(sim_matrix) if len(sim_matrix) == len(true_indices) else None
    for i in range(size):
        avg_rank += np.argwhere(sorted_indices[i] == true_indices[i])
        print(np.argwhere(sorted_indices[i] == true_indices[i]))
    return avg_rank / size


def mean_reciprocal_rank():
    print("Hello")


def median_rank():
    print("Hello")


def mean_average_precision():
    print("Hello")



testcase = np.array(
    [[0.02455722, 0.1507936,  0.35832496, 0.99897552, 0.78162878, 0.35370325, 0.83975505, 0.96822786, 0.21655323, 0.97689842],
     [0.11363342, 0.24912208, 0.29636562, 0.35155995, 0.29411583, 0.04967115, 0.0638789,  0.45134106, 0.9114714,  0.26269745],
     [0.01554174, 0.68342081, 0.04271253, 0.60616356, 0.71121971, 0.70394048, 0.24847037, 0.67655993, 0.87331004, 0.01168245],
     [0.41099609, 0.32924808, 0.37577786, 0.61338803, 0.81335371, 0.14777494, 0.22690066, 0.87883905, 0.49311936, 0.4395976 ],
     [0.47817871, 0.3319416,  0.41090754, 0.92866566, 0.2397549,  0.76909817, 0.85535081, 0.84577338, 0.25853286, 0.7426758 ],
     [0.63479745, 0.49362042, 0.93226137, 0.79895754, 0.10065838, 0.44211298, 0.40905317, 0.04747196, 0.89393958, 0.43807403],
     [0.22602446, 0.55747198, 0.3425144,  0.55212938, 0.18798087, 0.5354031, 0.65599647, 0.13325403, 0.90788474, 0.8081328 ],
     [0.05308029, 0.90290062, 0.19671464, 0.8270983,  0.75977151, 0.9606317, 0.56094346, 0.3132297,  0.78061087, 0.99156146],
     [0.85260693, 0.69077374, 0.3090848,  0.67289797, 0.35121438, 0.94652733, 0.78861944, 0.28010724, 0.65484997, 0.52046814],
     [0.15752759, 0.34159865, 0.86775922, 0.61466713, 0.89016877, 0.10684396, 0.31871215, 0.34062838, 0.40676347, 0.89294168]]
)

# y_true: [3, 8, 8, 7, 3, 2, 8, 9, 5, 9]


