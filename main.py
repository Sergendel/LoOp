import numpy as np
import pandas as pd
from math import erf
import matplotlib.pyplot as plt


class LoOp:
    def __init__(self, lmbda, K, file_path):

        self.L = lmbda
        self.K = K

        # Load data
        self.S = self.load_csv_data(file_path)

        # Initialize LoOP scores array
        self.LoOP = np.zeros((self.S.shape[0],))

        # Compute pdists (auxiliary array)
        self.pdists = self.calc_pdists()

        # Calculate normalization factor nPLOF
        self.nplof = self.calc_nplof()

        # Calculate LoOP scores
        self.calc_LoOP()

        # Plot LoOP scores
        self.plot_LoOP()

    def load_csv_data(self, file_path):
        df = pd.read_csv(file_path, encoding='utf8', header=None)
        S = df.to_numpy()
        return S

    def calc_pdists(self):
        pdist_S = np.zeros((self.S.shape[0],))
        for i_s in range(self.S.shape[0]):
            s = self.S[i_s, :]
            S_s, _ = self.context_set(s, self.S, self.K)
            pdist_S[i_s] = self.pdist(self.L, s, S_s)
        return pdist_S

    @staticmethod
    def e_dist(o, S):
        return np.sqrt(np.sum((o - S)**2, axis=1))

    @staticmethod
    def pdist(l, o, S):
        # wrong implementation, squaring the sum directly, which is incorrect.:
        # return l * np.square(np.sum((o - S) ** 2) / S.shape[0])

        # Corrected: mean of squared distances then square root
        distances_squared = np.sum((o - S)**2, axis=1)
        mean_distance_squared = np.mean(distances_squared)
        return l * np.sqrt(mean_distance_squared)

    def context_set(self, o, S, K):
        dists = self.e_dist(o, S)
        idx = np.argsort(dists)[:K]
        S_o = S[idx, :]
        return S_o, idx

    def calc_plof(self, o):
        S_o, idx = self.context_set(o, self.S, self.K)
        pdist_S_o = self.pdist(self.L, o, S_o)
        E_s = np.mean(self.pdists[idx])
        plof = (pdist_S_o / E_s) - 1
        return plof

    def calc_nplof(self):
        # wrong implementation, incorrectly squared the normalization factor at the end:
        # E = np.sum([self.calc_plof(self.S[i_s, ...]) ** 2 for i_s in range(self.S.shape[0])])
        # return self.L * np.square(E / self.S.shape[0])

        # correct implementation
        sum_plof_squared = np.sum([self.calc_plof(self.S[i])**2 for i in range(self.S.shape[0])])
        return self.L * np.sqrt(sum_plof_squared / self.S.shape[0])

    def calc_LoOP(self):
        # wrong implementation,
        # The original LoOP definition doesn't explicitly clamp at zero;
        # instead, the erf function naturally handles negatives:
        # for i_o in range(self.S.shape[0]):
        #     o = self.S[i_o, ...]
        #     plof = self.calc_plof(o)
        #     self.LoOP[i_o] = max(0, erf(plof / (np.sqrt(2) * self.nplof)))

        for i_o in range(self.S.shape[0]):
            o = self.S[i_o, :]
            plof = self.calc_plof(o)
            loop_score = erf(plof / (np.sqrt(2) * self.nplof))
            self.LoOP[i_o] = loop_score

        print("LoOP Scores:", self.LoOP)

    def plot_LoOP(self):
        font_size = 20
        plt.rcParams.update({'font.size': font_size})

        # Red color indicates high outlier probability (LoOP >= 0.9)
        colors = ['b' if score < 0.9 else 'r' for score in self.LoOP]
        plt.scatter(self.S[:, 0], self.S[:, 1], marker='.', color=colors)
        plt.title("LoOP Anomaly Detection")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Parameters
    file_path = "data/updated_enhanced_data.csv"
    lmbda = 3  # lambda parameter (usually between 1-3)
    K = 5      # number of nearest neighbors

    # Run LoOP
    loop_detector = LoOp(lmbda, K, file_path)
