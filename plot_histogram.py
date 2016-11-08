import numpy as np
import matplotlib.pyplot as plt

# retrieve age column
age = np.genfromtxt("census_data/census_train.csv", delimiter=",", skip_header=1, usecols=[0])

# plot histogram of age
plt.hist(age, bins=20, normed=False)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count / people")
plt.show()
