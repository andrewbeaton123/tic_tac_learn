"""In order to determine the number of tests required,
 we can use the formula for the sample size in a binomial experiment.
   The formula is:

n = Z^2 * p * (1-p) / E^2

where: Z is the z-score (for a 95% confidence level, Z = 1.96), 
p is the estimated win rate (p = 0.02), 
E is the margin of error (E = 0.02, 
which is the increase in win rate you are looking for).

So,

n = 1.96^2 * 0.02 * (1-0.02) / 0.02^2
"""
import math

# Constants
Z = 1.96  # Z-score for 95% confidence
p = 0.45  # Estimated win rate
E = 0.01  # Margin of error

# Calculate sample size
n = math.ceil(Z**2 * p * (1-p) / E**2)

print(n)