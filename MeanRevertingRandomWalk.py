import numpy as np
import matplotlib.pyplot as plt

# Based on equation 4.12 from Paul Wilmott on Quantitative Finance

# Mean-reverting models are used to model a random variable that 'isn't going anywhere'
# So is often used to model interest rates

# Parameters for the Ornstein-Uhlenbeck process
nu = 1.5  # Long-term mean level
sigma = 0.15  # Volatility parameter
gamma = 0.1  # Reversion Rate
r0 = 2  # Initial value of the process
T = 1  # Total time
N = 1000  # Number of time steps
dt = T/N  # Time step size

# Time vector
t = np.linspace(0, T, N)

# Initialize the process
r = np.zeros(N)
r[0] = r0

# Simulate the process
for i in range(1, N):
    # The mean-reversion term
    mean_reversion = gamma * (nu - r[i-1]) * dt
    # The randomness term
    randomness = sigma * np.sqrt(dt) * np.random.normal()
    # Update the process
    r[i] = r[i-1] + mean_reversion + randomness

# Plot the process
plt.plot(t, r)
plt.xlabel('Time')
plt.ylabel('r(t)')
plt.title('Ornstein-Uhlenbeck Process Simulation')
plt.grid(True)
plt.ylim(0, max(r) * 1.1)
plt.show()