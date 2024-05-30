import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # acceleration due to gravity in m/s^2
alpha = 8.1e-3  # PM spectrum constant
rho = 1025  # Density of seawater in kg/m^3
knots_to_ms = 0.51444  # Conversion factor from knots to meters per second
U_knots = 10  # Wind speed in knots
U = U_knots * knots_to_ms  # Convert wind speed to m/s
N = 1000  # Number of frequency components
dt = 0.1  # Time step in seconds
T = 1000  # Total time duration in seconds
W = 10    # Width of the structure in meters

# Calculate the peak frequency
f_0 = g / U

# Frequency range
frequencies = np.linspace(0.01, 1.0, N)
df = frequencies[1] - frequencies[0]

# PM Spectrum calculation
S_f = (alpha * g**2 / frequencies**5) * np.exp(-5/4 * (f_0 / frequencies)**4)

# Generate random phases
random_phases = np.random.uniform(0, 2*np.pi, N)

# Amplitude of each frequency component
amplitudes = np.sqrt(2 * S_f * df)

# Time vector
time = np.arange(0, T, dt)
wave_elevation = np.zeros(len(time))

# Construct the wave elevation time series
for i in range(N):
    wave_elevation += amplitudes[i] * np.cos(2 * np.pi * frequencies[i] * time + random_phases[i])

# Force calculation using static approximation
force = 0.5 * rho * g * wave_elevation**2 * W

# Plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(time, wave_elevation)
plt.xlabel('Time (seconds)')
plt.ylabel('Wave Elevation (meters)')
plt.title('Wave Elevation over Time')

plt.subplot(1, 2, 2)
plt.plot(time, force)
plt.xlabel('Time (seconds)')
plt.ylabel('Force (Newtons)')
plt.title('Wave Force on Structure over Time')
plt.tight_layout()
plt.show()
