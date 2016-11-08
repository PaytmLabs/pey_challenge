import csv
import numpy as np
from sklearn.linear_model import LogisticRegression

# organize files
directory = "census_data/"
train_file = directory + "census_train_m.csv"
test_file = directory + "census_test_m.csv"
files = [[directory + "census_train.csv", train_file],
         [directory + "census_test.csv", test_file]]

# drop fnlwgt column
for input_file, output_file in files:
    with open(input_file, "r") as in_file, open(output_file, "w") as out_file:
        reader = csv.reader(in_file)
        writer = csv.writer(out_file)
        next(reader)  # skip header
        for row in reader:
            del row[2]
            writer.writerow(row)

logit = LogisticRegression()

# train the classifier
train_data = np.genfromtxt(train_file, delimiter=",")
train = train_data[:, :-1]
target = train_data[:, -1]
logit = logit.fit(train, target)

# run the test on the model
test_data = np.genfromtxt(test_file, delimiter=",")
sample = test_data[:, :-1]
expected = test_data[:, -1]
score = logit.score(sample, expected)

print("Accuracy of this model is {0:1.13f}\n".format(score))
