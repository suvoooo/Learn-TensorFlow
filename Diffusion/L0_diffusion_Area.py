'''
trying to figure out why Eq. 13 in 2006.11239 (stable diffusion)
is only important when the network prediction is really good
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Given mean and variance
mean_good = 88 / 255
mean_bad = 93/255

variance = 1e-4

# Standard deviation (sigma) is the square root of the variance
std_dev = np.sqrt(variance)

# Define the range for x values
x_g = np.linspace(mean_good - 3 * std_dev, mean_good + 3 * std_dev, 100)
x_b = np.linspace(mean_bad - 3 * std_dev, mean_bad + 3 * std_dev, 100)

# Calculate the corresponding probabilities (PDF) for each x
def pdf(x, mean, std):
  pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std**2))
  return pdf

good_pdf = pdf(x_g, mean_good, std_dev)
bad_pdf = pdf(x_b, mean_bad, std_dev)

### good preds
x_l_t = 86/255
x_h_t = 88/255



# Calculate the cumulative distribution function (CDF) for x_l and x_h
cdf_x_l_g = norm.cdf(x_l_t, mean_good, std_dev)
cdf_x_h_g = norm.cdf(x_h_t, mean_good, std_dev)

cdf_x_l_b = norm.cdf(x_l_t, mean_bad, std_dev)
cdf_x_h_b = norm.cdf(x_h_t, mean_bad, std_dev)

# Calculate the area under the curve between x_l and x_h
area_under_curve_g = cdf_x_h_g - cdf_x_l_g
area_under_curve_b = cdf_x_h_b - cdf_x_l_b

print(f"Area under the curve between x_l and x_h for good prediction: {area_under_curve_g:.4f}")
print(f"Area under the curve between x_l and x_h for bad prediction: {area_under_curve_b:.4f}")


# Plot the normal distribution
fig = plt.figure(figsize=(12, 5))
fig.add_subplot(121)
plt.plot(x_g, good_pdf, label='Normal Distribution', color='blue')
plt.fill_between(x_g, good_pdf, 0, where=((x_g >= x_l_t) & (x_g <= x_h_t)), color='black', alpha=0.5)

# Add vertical lines at x_l and x_h

plt.axvline(x_l_t, color='red', linestyle='--', label='x_l')
plt.axvline(x_h_t, color='green', linestyle='--', label='x_h')

plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Mean = %3.5f and Variance = 1e-4: Area = %3.5f'%(mean_good, area_under_curve_g))
plt.legend()
plt.grid(True)
fig.add_subplot(122)
plt.plot(x_b, bad_pdf, label='Normal Distribution', color='blue')
plt.fill_between(x_b, bad_pdf, 0, where=((x_b >= x_l_t) & (x_b <= x_h_t)), color='black', alpha=0.5)

# Add vertical lines at x_l and x_h

plt.axvline(x_l_t, color='red', linestyle='--', label='x_l')
plt.axvline(x_h_t, color='green', linestyle='--', label='x_h')

plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Mean = %3.5f and Variance = 1e-4: Area = %3.5f'%(mean_bad, area_under_curve_b))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

