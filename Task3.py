# -*- coding: utf-8 -*-
'''
Моделирование отражения гармонического сигнала от слоя диэлектрика
'''

import math
import numpy as np
import numpy.typing as npt
from typing import Optional, List
from abc import ABCMeta, abstractmethod
import pylab
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

class LayerContinuous:
    def __init__(self,
                 xmin: float,
                 xmax: Optional[float] = None,
                 eps: float = 1.0,
                 mu: float = 1.0,
                 sigma: float = 0.0):
        self.xmin = xmin
        self.xmax = xmax
        self.eps = eps
        self.mu = mu
        self.sigma = sigma


class LayerDiscrete:
    def __init__(self,
                 xmin: int,
                 xmax: Optional[int] = None,
                 eps: float = 1.0,
                 mu: float = 1.0,
                 sigma: float = 0.0):
        self.xmin = xmin
        self.xmax = xmax
        self.eps = eps
        self.mu = mu
        self.sigma = sigma


class Probe:
    '''
    Класс для хранения временного сигнала в пробнике.
    '''

    def __init__(self, position: int, maxTime: int):
        '''
        position - положение пробника (номер ячейки).
        maxTime - максимально количество временных шагов для хранения в пробнике.
        '''
        self.position = position

        # Временные сигналы для полей E и H
        self.E = np.zeros(maxTime)
        self.H = np.zeros(maxTime)

        # Номер временного шага для сохранения полей
        self._time = 0

    def addData(self, E: npt.NDArray[float], H: npt.NDArray[float]):
        '''
        Добавить данные по полям E и H в пробник.
        '''
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1


class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)


class SourcePlaneWave(metaclass=ABCMeta):
    @abstractmethod
    def getE(self, position, time):
        pass


class Source1D(metaclass=ABCMeta):
    '''
    Базовый класс для всех источников одномерного метода FDTD
    '''
    @abstractmethod
    def getE(self, time):
        '''
        Метод должен возвращать значение поля источника в момент времени time
        '''
        pass

    def getH(self, time):
        return 0.0


class SourceTFSF(Source1D):
    def __init__(self, source: SourcePlaneWave,
                 sourcePos: float,
                 Sc: float = 1.0,
                 eps: float = 1.0,
                 mu: float = 1.0):
        self.source = source
        self.sourcePos = sourcePos
        self.Sc = Sc
        self.eps = eps
        self.mu = mu
        self.W0 = 120.0 * np.pi

    def getH(self, time):
        return -(self.Sc / (self.W0 * self.mu)) * self.source.getE(self.sourcePos, time)

    def getE(self, time):
        return (self.Sc / np.sqrt(self.eps * self.mu)) * self.source.getE(self.sourcePos - 0.5, time + 0.5)


class GaussianPlaneWave(SourcePlaneWave):
    '''
    Источник, создающий дифференцированный гауссов импульс
    '''

    def __init__(self, magnitude, dg, wg, dt, eps=1.0, mu=1.0, Sc=1.0):
        '''
        magnitude - максимальное значение в источнике;
        dg - коэффициент, задающий начальную задержку гауссова импульса;
        wg - коэффициент, задающий ширину гауссова импульса.
        '''
        self.magnitude = magnitude
        self.dg = dg
        self.wg = wg
        self.dt = dt
        self.eps = eps
        self.mu = mu
        self.Sc = Sc

    def getE(self, m, q):
        return -np.exp(-(((q - (m * np.sqrt(self.eps * self.mu)) / (self.Sc))*self.dt - self.dg)
                     / (self.wg))**2)


class AnimateFieldDisplay:
    '''
    Класс для отображения анимации распространения ЭМ волны в пространстве
    '''

    def __init__(self,
                 dx: float,
                 dt: float,
                 maxXSize: int,
                 minYSize: float, maxYSize: float,
                 yLabel: str,
                 title: Optional[str] = None
                 ):
        '''
        dx - дискрет по простарнству, м
        dt - дискрет по времени, сек
        maxXSize - размер области моделирования в отсчетах.
        minYSize, maxYSize - интервал отображения графика по оси Y.
        yLabel - метка для оси Y
        '''
        self.maxXSize = maxXSize
        self.minYSize = minYSize
        self.maxYSize = maxYSize
        self._xList = None
        self._line = None
        self._xlabel = 'x, м'
        self._ylabel = yLabel
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'
        self._dx = dx
        self._dt = dt
        self._title = title

    def activate(self):
        '''
        Инициализировать окно с анимацией
        '''
        self._xList = np.arange(self.maxXSize) * self._dx

        # Включить интерактивный режим для анимации
        pylab.ion()

        # Создание окна для графика
        self._fig, self._ax = pylab.subplots(
            figsize=(10, 6.5))

        if self._title is not None:
            self._fig.suptitle(self._title)

        # Установка отображаемых интервалов по осям
        self._ax.set_xlim(0, self.maxXSize * self._dx)
        self._ax.set_ylim(self.minYSize, self.maxYSize)

        # Установка меток по осям
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)

        # Включить сетку на графике
        self._ax.grid()

        # Отобразить поле в начальный момент времени
        self._line, = self._ax.plot(self._xList, np.zeros(self.maxXSize))

    def drawProbes(self, probesPos: list[int]):
        '''
        Нарисовать пробники.

        probesPos - список координат пробников для регистрации временных
            сигналов (в отсчетах).
        '''
        # Отобразить положение пробника
        self._ax.plot(np.array(probesPos) * self._dx,
                      [0] * len(probesPos), self._probeStyle)

        for n, pos in enumerate(probesPos):
            self._ax.text(
                pos * self._dx,
                0,
                '\n{n}'.format(n=n + 1),
                verticalalignment='top',
                horizontalalignment='center')

    def drawSources(self, sourcesPos: list[int]):
        '''
        Нарисовать источники.

        sourcesPos - список координат источников (в отсчетах).
        '''
        # Отобразить положение пробника
        self._ax.plot(np.array(sourcesPos) * self._dx,
                      [0] * len(sourcesPos), self._sourceStyle)

    def drawBoundary(self, position: int):
        '''
        Нарисовать границу в области моделирования.

        position - координата X границы (в отсчетах).
        '''
        self._ax.plot([position * self._dx, position * self._dx],
                      [self.minYSize, self.maxYSize],
                      '--k')

    def stop(self):
        '''
        Остановить анимацию
        '''
        pylab.ioff()

    def updateData(self, data: npt.NDArray[float], timeCount: int):
        '''
        Обновить данные с распределением поля в пространстве
        '''
        self._line.set_ydata(data)
        time_str = '{:.5f}'.format(timeCount * self._dt * 1e9)
        self._ax.set_title(f'{time_str} нс')
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


def showProbeSignalsOne(probes: List[Probe], minYSize: float, maxYSize: float):
    '''
    Показать графики сигналов, зарегистрированых в датчиках.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''
    # Создание окна с графиков
    fig, ax = plt.subplots()

    # Настройка внешнего вида графиков
    ax.set_xlim(0, len(probes[0].E) * dt * 1e9)
    ax.set_ylim(minYSize, maxYSize)
    ax.set_xlabel('t, нс')
    ax.set_ylabel('Ez, В/м')
    ax.grid()
    time_list = np.arange(len(probes[0].E)) * dt * 1e9
    # Вывод сигналов в окно
    for probe in probes:
        ax.plot(time_list, probe.E)

    # Показать окно с графиками
    plt.show()


def sampleLayer(layer_cont: LayerContinuous, sampler: Sampler) -> LayerDiscrete:
    start_discrete = sampler.sample(layer_cont.xmin)
    end_discrete = (sampler.sample(layer_cont.xmax)
                    if layer_cont.xmax is not None
                    else None)
    return LayerDiscrete(start_discrete, end_discrete,
                         layer_cont.eps, layer_cont.mu, layer_cont.sigma)


def fillMedium(layer: LayerDiscrete,
               eps: npt.NDArray[np.float64],
               mu: npt.NDArray[np.float64],
               sigma: npt.NDArray[np.float64]):
    if layer.xmax is not None:
        eps[layer.xmin: layer.xmax] = layer.eps
        mu[layer.xmin: layer.xmax] = layer.mu
        sigma[layer.xmin: layer.xmax] = layer.sigma
    else:
        eps[layer.xmin:] = layer.eps
        mu[layer.xmin:] = layer.mu
        sigma[layer.xmin:] = layer.sigma


if __name__ == '__main__':
    # Используемые константы
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Скорость света в вакууме
    c = 299792458.0

    # Электрическая постоянная
    eps0 = 8.854187817e-12

    # Параметры моделирования
    # Частота сигнала, Гц
    f_Hz = 0.15e9


    # Дискрет по пространству в м
    dx = 2e-3

    wavelength = c / f_Hz
    Nl = wavelength / dx

    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в м
    maxSize_m = 9.0

    # Время расчета в секундах
    maxTime_s = 100e-9
    df = 1 / maxTime_s
    # Положение источника в м
    sourcePos_m = 5

    # Координаты датчиков для регистрации поля в м
    probesPos_m = [6]

    # Параметры слоев
    layers_cont = [LayerContinuous(0, eps=8.5, sigma=0.0)]

    # Скорость обновления графика поля
    speed_refresh = 150

    # Переход к дискретным отсчетам
    # Дискрет по времени
    dt = dx * Sc / c

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)

    # Положение источника в отсчетах
    sourcePos = sampler_x.sample(sourcePos_m)

    layers = [sampleLayer(layer, sampler_x) for layer in layers_cont]

    # Датчики для регистрации поля
    probesPos = [sampler_x.sample(pos) for pos in probesPos_m]
    probes = [Probe(pos, maxTime) for pos in probesPos]

    # Вывод параметров моделирования
    print(f'Число Куранта: {Sc}')
    print(f'Размер области моделирования: {maxSize_m} м')
    print(f'Время расчета: {maxTime_s * 1e9} нс')
    print(f'Координата источника: {sourcePos_m} м')
    print(f'Частота сигнала: {f_Hz * 1e-9} ГГц')
    print(f'Длина волны: {wavelength} м')
    print(f'Количество отсчетов на длину волны (Nl): {Nl}')
    probes_m_str = ', '.join(['{:.6f}'.format(pos) for pos in probesPos_m])
    print(f'Дискрет по пространству: {dx} м')
    print(f'Дискрет по времени: {dt * 1e9} нс')
    print(f'Координаты пробников [м]: {probes_m_str}')
    print()
    print(f'Размер области моделирования: {maxSize} отсч.')
    print(f'Время расчета: {maxTime} отсч.')
    print(f'Координата источника: {sourcePos} отсч.')
    probes_str = ', '.join(['{}'.format(pos) for pos in probesPos])
    print(f'Координаты пробников [отсч.]: {probes_str}')

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)


    # Проводимость
    sigma = np.zeros(maxSize)

    for layer in layers:
        fillMedium(layer, eps, mu, sigma)

    # Источник
    magnitude = 100.0
    A_max = 100
    F_max = 0.15e9
    w_g = np.sqrt(np.log(A_max)) / (np.pi * F_max)
    d_g = w_g * np.sqrt(np.log(A_max))

    signal = GaussianPlaneWave(magnitude, d_g, w_g, dt)
    source = SourceTFSF(signal, 0.0, Sc, eps[sourcePos], mu[sourcePos])

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)
    # Ez[1] в предыдущий момент времени
    oldEzLeft = Ez[1]
    # Ez[-2] в предыдущий момент времени
    oldEzRight = Ez[-2]
    # Расчет коэффициентов для граничных условий
    tempLeft = Sc / np.sqrt(mu[0] * eps[0])
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)
    tempRight = Sc / np.sqrt(mu[-1] * eps[-1])
    koeffABCRight = (tempRight - 1) / (tempRight + 1)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.1
    display_ymax = 2.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = AnimateFieldDisplay(dx, dt,
                                        maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,
                                        title='fdtd_dielectric')

    display.activate()
    display.drawSources([sourcePos])
    display.drawProbes(probesPos)
    for layer in layers:
        display.drawBoundary(layer.xmin)
        if layer.xmax is not None:
            display.drawBoundary(layer.xmax)

    for t in range(1, maxTime):

        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] += source.getH(t)
        Hy[sourcePos - 1] -= source.getH(t)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1: -1] + (Hy[1:] - Hy[: -1]) * Sc * W0 / eps[1: -1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] -= source.getE(t)

        # Граничные условия ABC первой степени
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]
        Ez[-1] = oldEzRight + koeffABCRight * (Ez[-2] - Ez[-1])
        oldEzRight = Ez[-2]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % speed_refresh == 0:
            display.updateData(display_field, t)

    display.stop()

    # Отображение сигнала, сохраненного в пробнике
    showProbeSignalsOne(probes, -2.1, 2.1)
    # Построение спектра
    plt.figure()
    #t = np.arange(500)
    sp = np.fft.fft(probes[0].E)
    freq = np.fft.fftfreq(maxTime)
    for i in range(len(freq)):
        if freq[i] < 0:
            sp[i] = 0
    plt.plot(freq / (dt * 1e9), abs(sp) / max(abs(sp)))
    plt.xlim(0, 0.6)
    plt.grid()
    plt.xlabel('f, ГГц')
    display.stop()
    plt.show()
