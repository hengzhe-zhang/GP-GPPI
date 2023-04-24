import numpy as np


# Function F_1
def f1(x1, x2, x3):
    return -x1 * x2 / x3 ** 2


# Function F_2
def f2(x1, x2, x3):
    return 30 * x1 * x3 / ((x1 - 10) * x2 ** 2)


# Generate noise (50 random input variables) for each data point
def generate_noise(num_points):
    return np.random.rand(num_points, 50)


# Generate training and test samples for F_1
num_train_f1 = 70
num_test_f1 = 30
x1_train_f1 = np.random.rand(num_train_f1)
x2_train_f1 = np.random.rand(num_train_f1)
x3_train_f1 = np.random.uniform(1, 2, num_train_f1)
y_train_f1 = f1(x1_train_f1, x2_train_f1, x3_train_f1)
noise_train_f1 = generate_noise(num_train_f1)

x1_test_f1 = np.random.rand(num_test_f1)
x2_test_f1 = np.random.rand(num_test_f1)
x3_test_f1 = np.random.uniform(1, 2, num_test_f1)
y_test_f1 = f1(x1_test_f1, x2_test_f1, x3_test_f1)
noise_test_f1 = generate_noise(num_test_f1)

# Generate training and test samples for F_2
num_train_f2 = 1000
num_test_f2 = 10000
x1_train_f2 = np.random.uniform(-1, 1, num_train_f2)
x2_train_f2 = np.random.uniform(1, 2, num_train_f2)
x3_train_f2 = np.random.uniform(-1, 1, num_train_f2)
y_train_f2 = f2(x1_train_f2, x2_train_f2, x3_train_f2)
noise_train_f2 = generate_noise(num_train_f2)

x1_test_f2 = np.random.uniform(-1, 1, num_test_f2)
x2_test_f2 = np.random.uniform(1, 2, num_test_f2)
x3_test_f2 = np.random.uniform(-1, 1, num_test_f2)
y_test_f2 = f2(x1_test_f2, x2_test_f2, x3_test_f2)
noise_test_f2 = generate_noise(num_test_f2)

# Concatenate noise with X arrays for F_1 training and test samples
X_train_f1 = np.column_stack((x1_train_f1, x2_train_f1, x3_train_f1, noise_train_f1))
X_test_f1 = np.column_stack((x1_test_f1, x2_test_f1, x3_test_f1, noise_test_f1))

# Concatenate noise with X arrays for F_2 training and test samples
X_train_f2 = np.column_stack((x1_train_f2, x2_train_f2, x3_train_f2, noise_train_f2))
X_test_f2 = np.column_stack((x1_test_f2, x2_test_f2, x3_test_f2, noise_test_f2))

print(X_train_f1, X_test_f1)
print(X_train_f1.shape, X_test_f1.shape)
print(X_train_f2, X_test_f2)
print(X_train_f2.shape, X_test_f2.shape)
