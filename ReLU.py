import matplotlib.pyplot as plt


def relu(x):
    return max([0,x])


def compose_relu(x, n):
    for _ in range(n):
        x = -0.2*relu(x)+1
    return x

x_vals = [0.1*i for i in range(-100, 100, 1)]
y_vals = [2*compose_relu(0.1*i, 2)-0.7*compose_relu(0.1*i, 2) for i in range(-100, 100, 1)]

plt.plot(x_vals, y_vals)
plt.show()