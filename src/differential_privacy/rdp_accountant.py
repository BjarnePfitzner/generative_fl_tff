import math

import tensorflow_privacy as tfp


class RDPAccountant:
    def __init__(self, q, z, N, max_eps, target_delta=1e-5, dp_type='central', rdp_orders=None):
        """Calculates the actual standard deviation of noise after minibatch/client averaging
        Args:
          q: the sampling rate - clients_per_round (for central DP), batch_size / dataset_size (for local DP).
          z: the noise multiplier.
          N: the number of clients (for central DP), the size of the dataset (for local DP).
        Returns:
          an RDP accountant that provides info about privacy spending
        """
        self.q = q
        self.z = z
        self.N = N
        self.max_eps = max_eps
        self.target_delta = target_delta
        self.dp_type = dp_type

        if rdp_orders is None:
            self.rdp_orders = [1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] + list(range(5, 64)) + [128, 256, 512]
        else:
            self.rdp_orders = rdp_orders

    def get_privacy_spending_for_n_steps(self, n_steps=None):    # todo possibly cache computed epoch - epsilon matches?
        """Calculates the the privacy spending for a given epoch/steps.
        Args:
          n_steps: the number of differentially private steps. Corresponds to
                        the global epoch (for central DP), the total number of local optimisation steps (for local DP).
        Returns:
          an RDP accountant that provides info about privacy spending
        """
        rdp = tfp.compute_rdp(self.q, noise_multiplier=self.z, steps=n_steps, orders=self.rdp_orders)
        eps, _, order = tfp.get_privacy_spent(self.rdp_orders, rdp, target_delta=self.target_delta)
        return eps, order

    def get_maximum_n_steps(self, base_n_steps=50, n_steps_increment=10, max_n_steps=None):
        cur_n_steps = base_n_steps
        cur_eps, order = self.get_privacy_spending_for_n_steps(cur_n_steps)
        while cur_eps < self.max_eps:
            cur_n_steps += n_steps_increment
            if max_n_steps is not None and cur_n_steps > max_n_steps:
                cur_eps, order = self.get_privacy_spending_for_n_steps(max_n_steps)
                return max_n_steps, cur_eps, order
            cur_eps, order = self.get_privacy_spending_for_n_steps(cur_n_steps)

        while cur_eps > self.max_eps:
            cur_n_steps -= 1
            cur_eps, order = self.get_privacy_spending_for_n_steps(cur_n_steps)

        return cur_n_steps, cur_eps, order

    def compute_actual_noise_std(self, S):
        """Calculates the actual standard deviation of noise after minibatch/client averaging
        Args:
          S: the L2 norm clip threshold.
        Returns:
          the standard deviation of the added noise
        """
        return (self.z * S) / max(math.floor(self.N * self.q), 1)

    def compute_distortion(self, S):
        """Calculates the distortion / privacy loss, following Geyer, et al. (2018). Differentially Private Federated Learning: A Client Level Perspective.
        Args:
          S: the L2 norm clip threshold.
        Returns:
          the distortion
        """
        return S**2 * self.compute_actual_noise_std(S)**2 / (self.q * self.N)