from pylab import legend, savefig, ion, arange, step, figure, plot, hist, xlabel, ylabel, title, bar, subplot, tight_layout, xlim, ylim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pickle
import numpy as np
import tikzplotlib

with open('Data.bin', 'rb') as file:
    data = pickle.load(file)

    clone_states = data[0]
    times = data[1]

    del data

time = []
populations = [[], [], []]


for t in times[0]:
    time.append(t)

for state in clone_states[0]:
    populations[0].append(state[0])
    populations[1].append(state[1])
    populations[2].append(state[2])

fig, graph = plt.subplots()
graph.plot(time, populations[0], '-', label='Clonotype 1')
graph.plot(time, populations[1], '-', label='Clonotype 2')
graph.plot(time, populations[2], '-', label='Clonotype 3')
graph.set_ylabel('Number of cells')
graph.set_xlabel('Time')
graph.set_facecolor('white')
graph.legend(loc='upper right')
# graph.set_ylim(0, 1)
# graph.set_xlim(1, max_level_value - initial_state[0] - initial_state[1])
graph.spines['bottom'].set_color('gray')
graph.spines['top'].set_color('gray')
graph.spines['right'].set_color('gray')
graph.spines['left'].set_color('gray')

fig.savefig('Gillespie.pdf')
tikzplotlib.clean_figure()
tikzplotlib.save('Gillespie.tex')

graph.clear()
fig.clear()
plt.close(fig='all')
