
import os
import dill
import numpy as np
import matplotlib.pyplot as plt
from TDAnderson_HF import TDAnderson, Plotter


#%% --- Running&Plotting Wrapper ---

class Ranger:

    def __init__(self, var, rng, num, args, args2=(1,1,1e-6), w=5):
        
        self.N, self.N_s, self.tol = args2
        self.w = w
        print(w)
        self.num = num
        
        self.HF = []
        if var == 'V0':
            self.V0 = np.linspace(rng[0], rng[1], num)
            self.V1, self.U, self.Q = args
            for i in range(num):
                self.HF.append(TDAnderson(self.N, self.Q, self.N_s, self.U,
                                          self.V1, self.V0[i], w=self.w, tol=self.tol))  
        elif var == 'V1':
            self.V1 = np.linspace(rng[0], rng[1], num)
            self.V0, self.U, self.Q = args
            for i in range(num):
                self.HF.append(TDAnderson(self.N, self.Q, self.N_s, self.U,
                                          np.array([[self.V1[i]]]), self.V0, w=self.w, tol=self.tol))    
        elif var == 'U':
            self.U = np.linspace(rng[0], rng[1], num)
            self.V0, self.V1, self.Q = args
            for i in range(num):
                self.HF.append(TDAnderson(self.N, self.Q, self.N_s, self.U[i],
                                          self.V1, self.V0, self.tol))
        elif var == 'Q':
            self.Q = np.arange(rng[0], rng[1]+1, step=int(np.ceil((rng[1] - rng[0])/num)),
                               dtype=int)
            print(self.Q)
            self.V0, self.V1, self.U = args
            for i in range(num):
                self.HF.append(TDAnderson(self.N, self.Q[i], self.N_s, self.U,
                                          np.array([[self.V1]]), self.V0, w=self.w, tol=self.tol))
            
        #Initiaise Plotter Class
        self.plot = []
        for i in range(num):
            self.plot.append(Plotter(self.HF[i]))
        print(var + ' chosen as variable')
        return
        
    
    def run_SCF(self):
        
        for i in range(len(self.HF)):
            #Generate Guess Density
            self.HF[i].Guess_Den()
            #Run and pray...
            self.HF[i].run_SCF(damping=0.99)
            print('SCF Run', i+1, 'of', self.num, 'completed')
        
    
    def save(self, filename):
        file = open(filename, 'wb')
        dill.dump(self, file)
        file.close()
        
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            datas = dill.load(f) 
        return datas
    
    
    def P_nt(self, ts, both=False):  
        fig, ax = plt.subplots()   
        for i in range(self.num):
            self.plot[i].P_Nt(ts, ax, label='Q=' + str(self.Q[i]), both=both)
            print('ok')
        ax.legend()
