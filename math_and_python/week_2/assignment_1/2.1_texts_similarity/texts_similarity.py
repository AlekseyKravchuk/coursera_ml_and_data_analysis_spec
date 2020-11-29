import re
from collections import defaultdict
from collections import Counter
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def main():
    in_file = 'sentences.txt'
    weights_file = 'weights.csv'
    solution_file = 'solution_week2_1.txt'
    uw = defaultdict(int)  # uw stands for unique words, defaultdict, contains indexed unique words, val - index
    cnt = []  # list of Counter() objects (per line basis); each of them contains word as key and it's count as value
    indices = 0

    with open(in_file, 'r') as f:
        lines = [[word for word in re.split('[^a-z]', line.lower()) if word] for line in f]
    for line in lines:
        cnt.append(Counter(line))
        for word in line:
            if word not in uw:
                uw[word] = indices
                indices += 1
    uw = {value: key for (key, value) in uw.items()}  # swap keys and values so it would be {0: 'word1', 1: 'word2' ...}
    occurrences = [cnt[i][uw[j]] for i in range(len(cnt)) for j in range(len(uw))]
    weights = np.array(occurrences, dtype=np.uintc).reshape((len(cnt), len(uw)))

    # store weights matrix in *.csv file
    pd.DataFrame(weights).to_csv(weights_file, header=list(uw.values()), index=None)

    cos_distances = cdist(weights[:1, :], weights[1:, :], metric='cosine').flatten()
    indices = np.argpartition(cos_distances, 2)[:2] + 1  # index 0 corresponds to cosine(weights[0, :], weights[1, :])
    indices.sort()
    with open(solution_file, 'w') as f:
        f.write(' '.join([str(entry) for entry in list(indices)]))
    # with open(solution_file, 'w') as f:
    #     np.savetxt(f, indices, delimiter=' ', fmt='%2s', newline=' ')


if __name__ == '__main__':
    main()
