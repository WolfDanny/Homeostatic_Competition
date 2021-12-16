from pylab import legend,savefig,ion,arange,step,figure,plot,hist,xlabel,ylabel,title,bar,subplot,tight_layout,xlim,ylim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pickle
import numpy as np

states_number = 3

def position(state):
    """Function to obtain the position in the coefficient matrix.

    Returns the position in the coefficient matrix corresponding to the state specified, if the state exceeds maximum number of cells returns -1.

    Arguments:
        state - State list (number of cells per clonotype).
    """

    for i in range(N):
        if state[i] > eta or state[i] <= 0:
            return -1

    place = 0
    for i in range(len(state)):
        place += (state[i] - 1) * (eta ** i)

    return place

with open('Data.bin','rb') as file:
    data = pickle.load(file)

    Solutions = data[0]
    prob = data[1]
    N = data[2]
    eta = data[3]
    mu = data[4]
    phi = data[5]
    gamma = data[6]

    del data


#solG = [[0 for _ in range(len(Solutions))] for _ in range(states_number)]
#
#for i in range(states_number):
#    with open('Data-'+str(i)+'.bin','rb') as file:
#        data = pickle.load(file)
#
#        absoprtion_times = data[0]
#        division_number = data[1]
#        N = data[2]
#        eta = data[3]
#        mu = data[4]
#        phi = data[5]
#        gamma = data[6]
#        initial_state = data[7]
#        realisations = data[8]
#
#        for j in range(len(solG[0])):
#            for k in range(realisations):
#                solG[i][j] += absoprtion_times[k][j]
#            solG[i][j] = solG[i][j] / float(realisations)
#
#        del data



#%% Graph of time to extinction as a function of the probability

pos1 = position([5,5,5])
sol1 = []
for i in range(len(Solutions)):
    sol1.append(Solutions[i][pos1])


pos2 = position([10,10,10])
sol2 = []
for i in range(len(Solutions)):
    sol2.append(Solutions[i][pos2])

pos3 = position([20,20,20])
sol3 = []
for i in range(len(Solutions)):
    sol3.append(Solutions[i][pos3])

pos4 = position([30,30,30])
sol4 = []
for i in range(len(Solutions)):
    sol4.append(Solutions[i][pos4])

fig = plt.figure()
graph = fig.add_subplot(111)
#graph.set_title('$\eta=${0}, $\mu=${1}'.format(eta,mu))
#for i in range(states_number):
#    plot(prob,[n for n in solG[i]],'x',color='black')
plot(prob, sol1, '.-', label='(5,5,5)') #,fillstyle='none'
plot(prob, sol2, '.:', label='(10,10,10)')
plot(prob, sol3, '.-.', label='(20,20,20)')
plot(prob, sol4, '.--', label='(30,30,30)')
ylabel('$\hat{\\tau}_{n_{1},n_{2},n_{3}}}$')
xlabel('$p_{1,2}$')
xlim(0,1)
legend(loc='best')
#legend(loc='upper right')
savefig(f'Graph-N{N}-eta{eta}-mu{int(mu)}.pdf')


#%% 3D plot

#fig = plt.figure()
#graph = fig.add_subplot(111,projection='3d')
#graph.set_title('Mean time to absorption')
#
#k = 19
#dx = dy = 1
#
#_x = np.arange(1,20)
#_y = np.arange(1,20)
#_xx,_yy = np.meshgrid(_x,_y)
#x = _xx.ravel()
#y = _yy.ravel()
#z = np.zeros_like(x)
#
#dz = []
#for i in range(len(x)):
#    pos = position([x[i],y[i]])
#    dz.append(Solutions[k][pos])
#    
##bound = min(dz) - 1
##for i in range(len(z)):
##    z[i] += bound
##    dz[i] -= bound
#
#graph.set_xlim3d([1,20])
#graph.set_xlabel('$n_{1}$')
#
#graph.set_ylim3d([1,20])
#graph.set_ylabel('$n_{2}$')
#
#graph.set_zlim3d([0,50])
#graph.set_zlabel('$\hat{\\tau}_{n_{1},n_{2}}}}$')
#
#graph.bar3d(x,y,z,dx,dy,dz,color='green',zsort='average')
#
#savefig('3dplot{}.png'.format(k))
#plt.show()


#%% Animated 3D plot

#def gen():
#    for i in range(0,21):
#        yield i
            
#def update_plot(value,x,y,z,dx,dy):
#    dz = []
#    for i in range(len(x)):
#        pos = position([x[i],y[i]])
#        dz.append(Solutions[value][pos])
#        
#    graph.clear()
#    graph.set_title('$p=${0:.2f}'.format(prob[value]))
#    graph.set_xlim3d([1,eta])
#    graph.set_xlabel('$n_{1}$')
#
#    graph.set_ylim3d([1,eta])
#    graph.set_ylabel('$n_{2}$')
#
#    graph.set_zlim3d([0,50])
#    graph.set_zlabel('$\hat{\\tau}_{n_{1},n_{2}}}}$')
#    test = graph.bar3d(x,y,z,dx,dy,dz,color='green',zsort='average')
#    return test
#
#fig = plt.figure()
#graph = fig.add_subplot(111,projection='3d')
#
#dx = dy = 1
#_x = np.arange(1,eta)
#_y = np.arange(1,eta)
#_xx,_yy = np.meshgrid(_x,_y)
#x = _xx.ravel()
#y = _yy.ravel()
#z = np.zeros_like(x)
#
#ani = animation.FuncAnimation(fig,update_plot,fargs=(x,y,z,dx,dy),interval=50)
#
#ani.save('animation.html',dpi=200)
#plt.show()