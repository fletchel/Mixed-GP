import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import norm
from numpy.linalg import cholesky, solve, inv
from matplotlib import pyplot as plt
from math import log
from viz import viz_points, viz_interval
import generate_data
from copy import deepcopy

class GaussianProcess:

    def __init__(self, l=1, noise=1, dim=1, amplitude=1):

        # set hyperparameters

        self.l = l
        self.noise = noise  # noise variance (i.e. n^2 !!)
        self.dim = dim  # dimension of data
        self.amplitude = amplitude

        self.mu = None
        self.tau = None

        self.train_data = {'interval':None, 'ordinal':None, 'point':None} # data of the form [x1,x2,...,xd,y] OR [x1,..,xd,a,b]
        self.all_x = None

        self.saved_mu = None
        self.saved_tau = None
        self.saved_Sigma = None
        self.saved_train_data = None
        self.saved_all_x = None

    def add_train_data(self, data, data_type):

        assert isinstance(data, np.ndarray)
        assert data_type in ['point', 'ordinal', 'interval']
        assert (data_type in ['point', 'ordinal']) == (data.shape[1] == self.dim + 1)  # if data is point/ordinal, ensure right size
        assert (data_type == "interval") == (data.shape[1] == self.dim + 2) # same for interval

        if self.train_data[data_type] is not None:

            self.train_data[data_type] = np.append(self.train_data[data_type], data, axis=0)

        else:

            self.train_data[data_type] = data

        validkeys = [k for k in self.train_data if self.train_data[k] is not None]

        self.all_x = [x[0:self.dim] for k in validkeys for x in self.train_data[k]]

    def temp_EP(self, datapoint, data_type, num_iters):

        self.saved_train_data = deepcopy(self.train_data)
        self.saved_all_x = self.all_x

        self.add_train_data(datapoint, data_type)
        self.Full_EP(num_iters, temp=True)


    def kernel(self, X1, X2):

        X1 = X1.reshape(-1, self.dim)
        X2 = X2.reshape(-1, self.dim)
        distances = np.square(distance_matrix(X1, X2)).astype("float")

        return self.amplitude * np.exp(-distances/(2*self.l**2))


    def interval_EP(self, max_iter=10):

        data = self.train_data["interval"]
        X = data[:, 0:self.dim]
        a = data[:, self.dim]
        b = data[:, self.dim+1]

        n = len(data)

        tau = np.zeros(n)
        v = np.zeros(n)
        K = self.kernel(X, X)
        Sigma = self.kernel(X, X)

        mu = np.zeros(n)

        converged = False
        iteration_count = 0

        while not converged and iteration_count < max_iter:

            iteration_count += 1
            prev_mu = mu

            for i in range(n):

                tau_cav = Sigma[i, i]**-1 - tau[i]
                v_cav = mu[i]*(Sigma[i, i]**-1) - v[i]

                mu_cav = v_cav/tau_cav
                sig_cav = 1/tau_cav

                z_a = (mu_cav - a[i]) / ((self.noise ** (1 / 2)) * (1 + (sig_cav / self.noise)) ** (1 / 2))
                z_b = (mu_cav - b[i]) / ((self.noise ** (1 / 2)) * (1 + (sig_cav / self.noise)) ** (1 / 2))
                Z = norm.cdf(z_a) - norm.cdf(z_b)

                mu_hat = mu_cav + ((sig_cav/(Z*(self.noise**0.5)))*(1/(1+(sig_cav/self.noise))**0.5))*(norm.pdf(z_a) - norm.pdf(z_b))
                var_hat = sig_cav + (sig_cav**2)*(1/(Z*(self.noise+sig_cav)))*(z_b*norm.pdf(z_b) - z_a*norm.pdf(z_a) - (1/Z)*(norm.pdf(z_a) - norm.pdf(z_b))**2)

                delta_tau = (1/var_hat) - tau_cav - tau[i]
                tau[i] = tau[i] + delta_tau

                v[i] = (mu_hat/var_hat) - v_cav
                temp_sig = Sigma[:,i].reshape(-1, n)

                Sigma = Sigma - (temp_sig.T @ temp_sig)*(delta_tau**-1 + Sigma[i,i])**-1

                mu = Sigma @ v

            S_tilde = tau*np.identity(n)
            L = cholesky(np.identity(n) + S_tilde**0.5 @ K @ S_tilde**0.5)


            V = solve(L, S_tilde**0.5 @ K)
            Sigma = K - (V.T @ V)

            mu = Sigma @ v


            if sum(np.square(mu - prev_mu))**0.5 < 0.01:

                converged = True

        print("converged in " + str(iteration_count) + " iterations")

        self.mu = mu
        self.tau = tau

    def Full_EP(self, max_iter=10, temp=False):

        # temp used for entropy estimation

        if temp:

            self.saved_mu = self.mu
            self.saved_tau = self.tau
            self.saved_Sigma = self.Sigma

        all_data = []
        data_types = []
        if self.train_data["interval"] is not None:

            a = self.train_data["interval"][:, self.dim]
            b = self.train_data["interval"][:, self.dim+1]

            for l in self.train_data["interval"]:
                all_data.append(l)

            data_types = data_types + ["interval"]*(len(self.train_data["interval"]))

        if self.train_data["point"] is not None:

            for l in self.train_data["point"]:
                all_data.append(l)
            data_types = data_types + ["point"]*(len(self.train_data["point"]))

        if self.train_data["ordinal"] is not None:

            for l in self.train_data["ordinal"]:

                all_data.append(l)

            data_types = data_types + ["ordinal"] * (len(self.train_data["ordinal"]))


        X = np.array(self.all_x)
        n = len(X)

        tau = np.zeros(n)
        v = np.zeros(n)
        K = self.kernel(X, X)
        Sigma = self.kernel(X, X)

        mu = np.zeros(n)

        converged = False
        iteration_count = 0

        while not converged and iteration_count < max_iter:

            iteration_count += 1
            prev_mu = mu

            for i in range(n):

                if data_types[i] == "interval":

                    tau_cav = Sigma[i, i]**-1 - tau[i]
                    v_cav = mu[i]*(Sigma[i, i]**-1) - v[i]

                    mu_cav = v_cav/tau_cav
                    sig_cav = 1/tau_cav

                    z_a = (mu_cav - a[i]) / ((self.noise ** (1 / 2)) * (1 + (sig_cav / self.noise)) ** (1 / 2))
                    z_b = (mu_cav - b[i]) / ((self.noise ** (1 / 2)) * (1 + (sig_cav / self.noise)) ** (1 / 2))
                    Z = norm.cdf(z_a) - norm.cdf(z_b)

                    mu_hat = mu_cav + ((sig_cav/(Z*(self.noise**0.5)))*(1/(1+(sig_cav/self.noise))**0.5))*(norm.pdf(z_a) - norm.pdf(z_b))
                    var_hat = sig_cav + (sig_cav**2)*(1/(Z*(self.noise+sig_cav)))*(z_b*norm.pdf(z_b) - z_a*norm.pdf(z_a) - (1/Z)*(norm.pdf(z_a) - norm.pdf(z_b))**2)

                    delta_tau = (1/var_hat) - tau_cav - tau[i]
                    tau[i] = tau[i] + delta_tau

                    v[i] = (mu_hat/var_hat) - v_cav
                    temp_sig = Sigma[:,i].reshape(-1, n)

                    Sigma = Sigma - (temp_sig.T @ temp_sig)*(delta_tau**-1 + Sigma[i,i])**-1

                    mu = Sigma @ v

                    self.Sigma = Sigma
                    self.mu = mu

                elif data_types[i] == "point":

                    cur_data = all_data[i]

                    y = cur_data[self.dim]
                    tau_cav = Sigma[i, i] ** -1 - tau[i]
                    v_cav = mu[i] * (Sigma[i, i] ** -1) - v[i]

                    mu_cav = v_cav / tau_cav
                    sig_cav = 1 / tau_cav

                    mu_hat = (sig_cav*y + self.noise*mu_cav)/(sig_cav + self.noise)
                    var_hat = 1/((1/sig_cav) + (1/self.noise))
                    Z = (2*3.14159*var_hat)**0.5

                    delta_tau = (1 / var_hat) - tau_cav - tau[i]


                    #delta_tau = max(0, 0.000001)
                    tau[i] = tau[i] + delta_tau


                    v[i] = (mu_hat / var_hat) - v_cav
                    temp_sig = Sigma[:, i].reshape(-1, n)

                    Sigma = Sigma - (temp_sig.T @ temp_sig) * (delta_tau ** -1 + Sigma[i, i]) ** -1

                    mu = Sigma @ v

                    self.Sigma = Sigma
                    self.mu = mu

            S_tilde = tau*np.identity(n)
            L = cholesky(np.identity(n) + S_tilde**0.5 @ K @ S_tilde**0.5)


            V = solve(L, S_tilde**0.5 @ K)
            Sigma = K - (V.T @ V)

            mu = Sigma @ v

            self.Sigma = Sigma
            self.mu = mu


            if sum(np.square(mu - prev_mu))**0.5 < 0.01:

                converged = True

        #print("converged in " + str(iteration_count) + " iterations")

        self.mu = mu
        self.tau = tau


    def predict_mean(self, x):

        x = np.array(x).reshape(-1, self.dim)

        k_star = self.kernel(x, np.array(self.all_x))
        K = self.kernel(np.array(self.all_x), np.array(self.all_x))
        S = self.tau*np.identity(len(self.tau))
        v = S @ self.mu

        mean = k_star @ inv((K + inv(S))) @ inv(S) @ v

        return mean

    def predict_var(self, x):

        x = np.array(x).reshape(-1, self.dim)
        k_star_star = self.kernel(x, x)

        k_star = self.kernel(x, np.array(self.all_x))

        K = self.kernel(np.array(self.all_x), np.array(self.all_x))
        S = self.tau * np.identity(len(self.tau))

        var = k_star_star - k_star @ inv(K + inv(S)) @ k_star.T

        return var.diagonal()


    def visualise_GP(self, viz_train_data=False, viz_variance=False):  # only works for 1-d

        assert self.dim == 1

        all_x_unrolled = [a[0] for a in self.all_x]
        min_x = min(all_x_unrolled) - 2
        max_x = max(all_x_unrolled) + 2

        viz_grid = np.linspace(min_x, max_x, num=200)

        predict_mu = self.predict_mean(viz_grid)
        plt.plot(viz_grid, predict_mu, color='blue')

        if viz_variance:

            predict_var = self.predict_var(viz_grid)
            std = predict_var**0.5

            plt.fill_between(viz_grid, predict_mu-std, predict_mu+std, alpha=0.6)

        plt.show()

    def max_var(self, x):

        predict_var = self.predict_var(x)

        m = np.argmax(predict_var)

        return m, x[m]

    def reverse_temp(self):

        self.Sigma = self.saved_Sigma
        self.mu = self.saved_mu
        self.tau = self.saved_tau
        self.train_data = deepcopy(self.saved_train_data)
        self.all_x = self.saved_all_x


    def post_point_variance(self, x, N):

        mu = self.predict_mean(x)
        var = self.predict_var(x)

        samples = np.random.normal(mu, var, N)

        estimate = 0

        for i in range(len(samples)):

            self.temp_EP(np.array([x, float(samples[i])]).reshape(1,2), "point", 8)

            estimate += self.predict_var(x)

            self.reverse_temp()

        estimate = estimate/N

        return estimate

    def post_interval_variance(self, x, N):

        mu = float(self.predict_mean(x))
        var = float(self.predict_var(x))

        samples = np.array([generate_data.generate_interval(mu, var**0.5, l_bounds=[var**0.5, 3*var**0.5]) for i in range(N)])

        estimate = 0

        for i in range(len(samples)):

            self.temp_EP(np.array([x, float(samples[i][0]), float(samples[i][1])]).reshape(1,3), "interval", 8)

            estimate += self.predict_var(x)

            self.reverse_temp()

        estimate = estimate/N

        return estimate

    def var_reduction_calc(self, costs, x, N):

        # costs = [point_cost, int_cost, ord_cost]
        cur_var = self.predict_var(x)
        adj_point = (1 - (self.post_point_variance(x, N)/cur_var))/costs[0]        # percentage reduction in variance per unit cost
        adj_int = (1 - (self.post_interval_variance(x, N)/cur_var))/costs[1]
        adj_ord = 0

        if adj_point > adj_int and adj_point > adj_ord:

            return "point", adj_point

        elif adj_int > adj_point and adj_int > adj_ord:

            return "interval", adj_int

        else:

            return "ordinal", adj_ord



    def active_learning(self, x_grid, f):

        arg, x = self.max_var(x_grid)

        d_type, adj = self.var_reduction_calc([1.2,1], x, 100)

        if d_type == "point":

            point = np.random.normal(f[arg], self.noise)
            self.add_train_data(np.array([x, point]).reshape(1,2), "point")

        elif d_type == "interval":

            interval = generate_data.generate_interval(f[arg], self.noise**0.5)
            self.add_train_data(np.array([x, interval[0], interval[1]]).reshape(1,3), "interval")

        self.Full_EP(10)
        print(d_type)
        print(x)
        print("\n")



