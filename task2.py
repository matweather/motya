import numpy
import requests
from scipy import special, constants
from numpy import arange, abs, sum, round
from matplotlib import pyplot as plt
import os
import re

# check dir 'result'
os.mkdir('result') if not os.path.isdir('result') else print('Уже есть такая директрория')

# parse txt
data = requests.get('https://jenyay.net/uploads/Student/Modelling/task_02_01.txt').text.split('\n')
for string in data:
    if string.startswith('16'):
        # Шаблон с использованием регулярных выражений
        pattern = re.compile(r'D=(.*?); fmin=(.*?); fmax=(.*?)$')

        # Метод findall для извлечения значений
        matches = pattern.findall(string)
        # Присвойте значения переменным
        D = float(matches[0][0])
        f_min = float(matches[0][1])
        f_max = float(matches[0][2])

# const
n_end = 25
f_step = 10000000
r = D / 2
f_arange = arange(f_min, f_max, f_step)
wavelength_arange = constants.c / f_arange
k_arange = 2 * constants.pi / wavelength_arange


# h
def f4(n, x):
    return special.spherical_jn(n, x) + 1j * special.spherical_yn(n, x)


# b
def f3(n, x):
    return (x * special.spherical_jn(n - 1, x) - n * special.spherical_jn(n, x)) / (x * f4(n - 1, x) - n * f4(n, x))


# a
def f2(n, x):
    return special.spherical_jn(n, x) / f4(n, x)


# Radar Cross Section(RCS) - ЭПР
rcs_arange = (wavelength_arange ** 2) / numpy.pi * (abs(sum([((-1) ** n) * (n+0.5) * (f3(n, k_arange * r) - f2(n, k_arange * r)) for n in range(1, n_end)], axis=0)) ** 2)

# File
with open('result/data.txt', 'w', encoding='utf8') as file:
    file.write('Частота,Гц    ЭПР\n')
    for f, rcs in zip(f_arange, rcs_arange):
        file.write(f'{f}    {rcs}\n')

# Graphic
plt.plot(f_arange, rcs_arange)
plt.xlabel("$f, Гц$")
plt.ylabel(r"$\sigma, м^2$")
plt.grid()
# Увеличение отступов
plt.tight_layout()
plt.show()

