# imports
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

# Initialise variables
asset = 100
drift = 0.1
volatility = 0.05
timestep = 0.01

time = 0
current_price = asset
iterations = 100

# Lists to store time and price data
times = []
prices = []

# Random Walk loop
for n in range(1, iterations + 1):
    # Append current time and price to the lists
    times.append(time)
    prices.append(current_price)
    
    # Print time and price values
    print(time, current_price)

    # Generate random number
    randomNum = sum(random.random() for _ in range(12)) - 6

    # Random Walk simulation from Paul Wilmott on Quantitative Finance
    current_price = current_price * (1 + (drift * timestep) + (volatility * math.sqrt(timestep) * randomNum))

    # Increment time and round to avoid floating point errors accumalating.
    time += timestep
    time = round(time, 2)

# Plot the graph
plt.plot(times, prices, marker='o') # Plot line graph with circular markers
plt.title('Random Walk Simulation over Time') # Title
plt.xlabel('Time') # X axis label
plt.ylabel('Asset Price') # Y axis label
plt.grid(True) # Display grid lines
plt.show() # Display the graph
