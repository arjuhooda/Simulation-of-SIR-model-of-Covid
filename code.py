import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('dark_background')
from matplotlib.animation import FuncAnimation as animation
import random
import os
 
inf_col = 'red'
rem_col = 'green'
ded_col = 'yellow'
sus_col = 'blue'
 
infection_radius = 0.6
recovery_rate = 0.0005
death_rate = .0001
stage_dimension = 300 # MxM square stage
 
init = [1500, # population size
        10, # infected
        0, #recovered
        0] #dead
 
maskBool=False
maskMod = 0.55
 
saniBool=False
saniMod = 0.35
 
SDbool=False
SDmod = 6
 
class Sim():
 
    def __init__(self,init,maskBool,saniBool,SDbool):
        #init = [N,inf,rec,dead]
        self.N = init[0]
        suc = self.N-init[1]-(init[2]+init[3])      #susceptibles
        self.cols = np.repeat([sus_col,inf_col,rem_col,ded_col],    #colors
                              [suc,init[1],init[2],init[3]])
        self.low,self.high = -stage_dimension/2,stage_dimension/2
        self.pos = np.random.uniform(self.low,self.high,size=(2,self.N))
        self.stream = self.data_stream()
        self.fig,self.ax = plt.subplots()
        self.ani = animation(self.fig, self.update, interval=80, 
                                          init_func=self.setup, blit=True)
 
    def setup(self):
        p,c= next(self.stream)
        self.scat = self.ax.scatter(x=p[0,:], y=p[1,:], c=c.T)
        return self.scat,
    
    def arg_within_radius(self,inf,susceptible,r):
        if(maskBool):
            r=r*maskMod
        dist = np.sqrt(((inf-susceptible)**2).sum(axis=1))
        return np.argwhere(dist<r).ravel()
    
    def data_stream(self):
        while True:
            removed = 0
            #move points
            jitter = 0.5*np.random.normal(0,1,size=(2,self.N))
            if(SDbool):
                jitter=jitter/SDmod
            self.pos[0:2,:]+=jitter
 
            #infect and recover---------
            #get infected people
            inf = np.argwhere(self.cols == inf_col).ravel()
            inf_people = self.pos[:,inf].T
            for i in inf_people:
                suc = np.argwhere(self.cols == sus_col).ravel()
                suc_people = self.pos[:,suc].T
                if(saniBool):
                    risk= random.random(0,1)
                    if(risk >saniMod):
                        infect = self.arg_within_radius(i,suc_people,infection_radius)
                    else:
                        pass
                else:
                    infect = self.arg_within_radius(i,suc_people,infection_radius)  
                infect_index = suc[infect]
                self.cols[infect_index] = inf_col
            
            #adjust out of bounds
            x1=np.where(self.pos[0:2,:]<self.low)
            x2=np.where(self.pos[0:2,:]>self.high)
            self.pos[0:2,:][x1],self.pos[0:2,:][x2] = self.low,self.high
 
            #yield [self.pos,self.cols]
            r = np.random.uniform(0,1, size = inf.size)
            rec = np.argwhere(r<recovery_rate).T
            recovered_idx = inf[rec]
            self.cols[recovered_idx] = rem_col
            d = np.random.uniform(0,1, size = (inf.size))
            dead = np.argwhere(d<death_rate).T
            dead_idx = inf[dead]
            self.cols[dead_idx] = ded_col
            ded=np.argwhere(self.cols == ded_col).ravel()
            ded_people = self.pos[:,dead].T
            for dd in ded_people:
                removed +=1
            yield self.pos,self.cols
            
    def update(self,i):
        # update the scatter collection
        data,c= next(self.stream)
        self.scat.set_offsets(data[0:2,:].T)    #change positons
        self.scat.set_sizes(np.zeros(self.N)+10)    # change size
        self.scat.set_color(c.T)    #change color
        return self.scat,
 
S = Sim(init,maskBool=False,saniBool=False, SDbool=False)
fig = plt.gcf()
fig.canvas.set_window_title('City 1')
 
S2 = Sim(init,maskBool=True,saniBool=True,SDbool=True)
fig = plt.gcf()
fig.canvas.set_window_title('City 2')
plt.show()
