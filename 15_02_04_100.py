import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

p_train = pd.read_csv('train.txt', header=None, sep=' ', dtype='float64')
train_arr = p_train.values
len_train = train_arr[:, 0].size

class_1 = []
class_2 = []

for i in range(len_train):
    if train_arr[i, 2] == 1:
        class_1.extend([train_arr[i, 0:2]])
    else:
        class_2.extend([train_arr[i, 0:2]])

class_1 = np.array(class_1)
class_2 = np.array(class_2)

x1 = class_1[:, 0]
y1 = class_1[:, 1]
x2 = class_2[:, 0]
y2 = class_2[:, 1]

plt.scatter(x1, y1, color='red', marker='+')
plt.scatter(x2, y2, color='green', marker='*')
plt.show()


def phi_function(m, n):
    return np.array([m ** 2, n ** 2, m * n, m, n, 1])


len_1 = class_1[:, 0].size
len_2 = class_2[:, 0].size

phi_1 = []
phi_2 = []

# determining phi function
for i in range(len_1):
    x = class_1[i, 0]
    y = class_1[i, 1]
    phi_1.extend([phi_function(x, y)])

phi_1 = np.array(phi_1)

for i in range(len_2):
    x = class_2[i, 0]
    y = class_2[i, 1]
    phi_2.extend([phi_function(x, y)])

phi_2 = np.array(phi_2)

# normalization
phi_2 = np.dot(phi_2, -1)

phi = np.concatenate((phi_1, phi_2))
length = phi[:, 0].size

while 1:
    print("select your initial weight: ")
    print("a. All ones\nb. All zeros\nc. Random initialization\nd. press s to stop")
    inp = input()
    if inp == 'a':
        weight = [1, 1, 1, 1, 1, 1]
    elif inp == 'b':
        weight = [0, 0, 0, 0, 0, 0]
    elif inp == 'c':
        np.random.seed(7)
        weight = np.random.randint(low=1, high=10, size=6)
    elif inp == 's':
        break

    learning_rate = []
    iteration = []
    weight_batch = []

    # batch process
    a = 0.1
    for k in range(10):
        initial_weight = weight

        for i in range(500):
            count = 0
            sum_weight = [0, 0, 0, 0, 0, 0]
            for j in range(length):
                if np.dot(phi[j, :], initial_weight) > 0:
                    count = count+1
                elif np.dot(phi[j, :], initial_weight) <= 0:
                    sum_weight += a*phi[j, :]
            initial_weight = sum_weight + initial_weight
            if count == length:
                break

        learning_rate.append(a)
        iteration.append(i + 1)
        weight_batch.append(initial_weight)
        a = a + 0.1

    # single process
    weight_single = []
    a = 0.1
    iteration_single = []
    for k in range(10):
        initial_weight = weight
        count = 0
        for i in range(500):
            sum_weight = [0, 0, 0, 0, 0, 0]
            for j in range(length):
                if np.dot(phi[j, :], initial_weight) > 0:
                    count = count+1
                    if count == length:
                        break
                elif np.dot(phi[j, :], initial_weight) <= 0:
                    sum_weight = initial_weight + a*phi[j, :]
                    count = 0
                    initial_weight = sum_weight
            if count == length:
                break
        iteration_single.append(i + 1)
        weight_single.append(initial_weight)
        a = a + 0.1

    # table creation

    dict = {'Learning rate': learning_rate, 'Many at a time': iteration, 'Weight Batch': weight_batch,
            'One at a time': iteration_single, 'Weight Single': weight_single}
    df = pd.DataFrame(dict)
    df.to_csv('table.csv', index=False)


    # bar plot

    labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, iteration, width, label='Many at a time')
    rects2 = ax.bar(x + width/2, iteration_single, width, label='one at a time')

    ax.set_ylabel('Iteration')
    ax.set_title('Comparison between Single processing vs Batch processing')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

# boundary line

w = weight_batch[0]

bx = np.linspace(-10, 10, 50)

final_y = np.zeros(50)
final_z = np.zeros(50)


def boundary_eqn1(p1, p2, p3):
    eqn = (-p2 - (p2 ** 2 - 4 * p1 * p3) ** 0.5) / (2 * p1)
    return eqn


def boundary_eqn2(p1, p2, p3):
    eqn = (-p2 + (p2 ** 2 - 4 * p1 * p3) ** 0.5) / (2 * p1)
    return eqn


for i in range(len(bx)):
    w1 = w[1]
    w2 = (w[2] * bx[i]) + w[4]
    w3 = (w[0] * bx[i]**2) + (w[3] * bx[i]) + w[5]

    final_y[i] = boundary_eqn1(w1, w2, w3)
    final_z[i] = boundary_eqn2(w1, w2, w3)


plt.scatter(x1, y1, color='red', marker='+')
plt.scatter(x2, y2, color='green', marker='*')
plt.plot(bx, final_y)
plt.plot(bx, final_z)
plt.show()
