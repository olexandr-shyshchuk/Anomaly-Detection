import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform, laplace, triang, loggamma, rayleigh, kstest


class NextStepWinsorize:
    def __init__(self):
        self.original_data = None
        self.winsorized_data = None
        self.xo = None

        self.previous_steps = np.array([])
        self.next_steps = np.array([])
        self.winsorized_steps = np.array([])
        self.__percentiles = np.array([5, 95])
        self.limits = np.array([])

        self.generated_data = np.array([])

    def fit_transform(self, data, N=13):
        self.start_points = data[:2]
        self.original_data = data
        wind = np.lib.stride_tricks.sliding_window_view(data, (3,))
        dxy = np.zeros((wind.shape[0], 2))
        dxy[:, 0] = wind[:, 1] - wind[:, 0]
        dxy[:, 1] = wind[:, 2] - wind[:, 1]
        del wind

        idxs = np.argsort(dxy[:, 0])[::-1]
        self.previous_steps = np.reshape(dxy[idxs], (dxy.shape[0] // N, N, 2))
        data_winsorized = np.zeros_like(self.previous_steps[:, :, 0])

        xo = np.array(())
        for i, (group, idx) in enumerate(zip(self.previous_steps, np.reshape(idxs, (dxy.shape[0] // N, N)))):

            if self.next_steps.size != 0:
                self.next_steps = np.vstack([self.next_steps, group[:, 1]])
            else:
                self.next_steps = np.array([group[:, 1]])

            xt = idx + 2

            low = np.percentile(group[:, 1], self.__percentiles[0])
            high = np.percentile(group[:, 1], self.__percentiles[1])

            if self.limits.size == 0:
                self.limits = np.array([low, high])
            else:
                self.limits = np.vstack([self.limits, np.array([low, high])])

            xt = xt[(group[:, 1] < low) | (group[:, 1] > high)]
            xo = np.append(xo, xt)

            data_winsorized[i] = np.clip(group[:, 1], low, high)

        self.winsorized_steps = data_winsorized
        temp = np.take(np.reshape(data_winsorized, (dxy.shape[0],)), np.argsort(idxs))
        self.winsorized_data = np.insert(temp, 0, data[:2])
        self.winsorized_data[1:] = np.cumsum(self.winsorized_data[1:])
        self.xo = xo
        return self.winsorized_data

    def show_winsorized(self):
        plt.plot(self.original_data, zorder=1, color='red', label='Original')
        plt.plot(self.winsorized_data, zorder=2, color='green', label='Winsorized')
        plt.legend()
        plt.show()

    def show_outliners(self):
        plt.plot(self.original_data, label='Data')
        if not np.all(self.original_data == self.winsorized_data):
            plt.scatter(x=self.xo, y=self.original_data[self.xo.astype(np.int16)], color='red', s=5, zorder=100,
                        label='Emmisions')
        plt.legend()
        plt.show()

    def generate_data(self, N=80, emmisions=True, prob=0.15):
        best_distributions = []
        best_params_all = []
        for i in range(self.next_steps.shape[0]):
            data = self.next_steps[i]

            best_distribution = None
            best_ks_pvalue = 0
            best_params = None

            distributions = [norm, expon, uniform, laplace, triang, loggamma, rayleigh]

            for distribution in distributions:
                params = distribution.fit(data)

                ks_statistic, ks_pvalue = kstest(data, distribution.name, params)

                if ks_pvalue > best_ks_pvalue:
                    best_distribution = distribution
                    best_ks_pvalue = ks_pvalue
                    best_params = params

            best_distributions.append(best_distribution)
            best_params_all.append(best_params)

        # --------------------------------------

        rnds = [np.random.RandomState(i**3) for i in range(6)]
        max_limits = self.previous_steps[:, :, 0][:, 0].copy()
        min_limits = self.previous_steps[:, :, 0][:, -1].copy()

        for i in range(1, len(max_limits)):
            mean = (max_limits[i] + min_limits[i - 1]) / 2
            max_limits[i] = min_limits[i-1] = mean

        max_limits[0] = np.inf
        min_limits[-1] = -np.inf

        # --------------------------------------

        generated_data = np.zeros(N)

        generated_data[:2] = self.original_data[:2]

        em_rnd = np.random.RandomState(1)

        change = generated_data[1] - generated_data[0]
        for i in range(2, N):
            mask = (min_limits < change) & (change<= max_limits)
            group = np.argmax(mask)

            change = best_distributions[group].rvs(*best_params_all[group], size=1, random_state=rnds[group])[0]

            if emmisions & (prob > em_rnd.uniform(0, 1)):
                while (change >= self.limits[group][0]) & (self.limits[group][1] <= change):
                    change = 2 * best_distributions[group].rvs(*best_params_all[group], size=1, random_state=rnds[group])[0]
            else:
                while (change <= self.limits[group][0]) | (self.limits[group][1] <= change):
                    change = best_distributions[group].rvs(*best_params_all[group], size=1, random_state=rnds[group])[0]

            generated_data[i] = generated_data[i-1] + change

        self.generated_data = generated_data
        return generated_data

    def show_generated_data(self):
        plt.plot(self.generated_data)
        plt.ylabel('USD')
        plt.title('Generated data')
        plt.show()


if __name__ == '__main__':
    data = np.array(pd.read_csv('price.csv')['Bitcoin'])
    wins = NextStepWinsorize()

    wins.fit_transform(data)
    wins.show_outliners()
    wins.show_winsorized()
