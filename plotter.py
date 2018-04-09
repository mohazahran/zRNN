'''
Created on Feb 21, 2018

@author: mohame11
'''
#from zLSTM_weights_to_outputs import *
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import ast


def plotByEpoch(C_epoch_states, epoch):
    fig = plt.figure(0)
    mpl.rcParams.update({'font.size': 13})
    #plt.xticks(list(np.arange(-0.5, 0.05, 0.5)), rotation='vertical')
    #plt.yticks(list(np.arange(-0.5, 0.05, 0.5)))
    plt.xlabel('ct[0]')
    plt.ylabel('ct[1]')
    fig.suptitle('The inner state (C) transitions at epoch#%d'%epoch, fontsize=9, fontweight='bold', horizontalalignment='center', y=.86)
    ax = plt.axes()
    mn, mx =  min(C_epoch_states[epoch].keys()), max(C_epoch_states[epoch].keys()) 
    ct_prev = [0.0,0.0]
    plt.plot(ct_prev[0], ct_prev[1], 'bo')
    for t in range(mn, mx+1):
    #for t in range(mn, 10):    
        ct = C_epoch_states[epoch][t]
        
        if t % 10 == 0:
            plt.plot(ct[0], ct[1], 'bo')
            plt.plot([ct_prev[0], ct[0]], [ct_prev[1], ct[1]], 'r')
        #ax.arrow(ct_prev[0], ct_prev[1], ct[0]-ct_prev[0], ct[1]-ct_prev[1], head_width=0.05, head_length=0.01, fc='k', ec='k')
            #ax.arrow(ct_prev[0], ct_prev[1], ct[0]-ct_prev[0], ct[1]-ct_prev[1])
        #ax.arrow(ct_prev[0], ct_prev[1], ct[0], ct[1], head_width=0.0005, head_length=0.005)
            ct_prev = ct
        #plt.show()
    
    plt.grid()                      
    plt.savefig('Cstates_transitions.pdf', bbox_inches='tight')
    plt.show()
    


def main():
    C_epoch_states = {}
    r = open('statesLog/C.txt', 'r')
    for line in r:
        parts = line.split(':')
        epoch = int(parts[0]); t = int(parts[1]);  c = ast.literal_eval(parts[2])
        if epoch in C_epoch_states:
            if t in C_epoch_states[epoch]: 
                print 'something is wrong at epoch=%d, t=%' % (epoch, t)
            else:
                C_epoch_states[epoch][t] = c
        else:
            C_epoch_states[epoch] = {t:c}
        
    plotByEpoch(C_epoch_states, -1)
        
        
    

if __name__ == '__main__':
    main()