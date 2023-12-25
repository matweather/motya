from math import sin, cos
from numpy import arange, vectorize
from matplotlib import pyplot as plt
import json
import os
from numpy import pi

# const
A = 10
x_start = 0
x_end = pi
x_step = 0.01


# our function
def f(x):
    return -sin(x) * sin(x ** 2 / pi) ** (2 * A)


# create arrays
x_arrays = arange(x_start, x_end, x_step)
f2 = vectorize(f)
y_arrays = f2(x_arrays)

# make plot
plt.plot(x_arrays, y_arrays)
plt.show()

# check dir 'result'
os.mkdir('result') if not os.path.isdir('result') else print('Уже есть такая директрория')

# add to file
with open('result/data.json', 'w') as file:
    result_dir = {'data': []}
    [result_dir['data'].append({'x': x, 'y': y}) for x, y in zip(x_arrays, y_arrays)]
    file.write(json.dumps(result_dir, indent=4))
