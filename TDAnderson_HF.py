import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
#from scipy.integrate import quad, quad_vec
from scipy.linalg import block_diag


#%% --- SCF Object ---

class TDAnderson:
    
    def __init__(self, N, Q, Ns, U, Vi, V0, w=10, tol=1e-5):
        
        Q = 2*Q+1
        self.N = int(N)
        self.Q = int(Q)
        self.Ns = int(Ns)
        self.Ef = 0.0 #U/2 + Vi[int(N/2), int(N/2)]
        self.U = U
        self.Vi = Vi
        self.V0 = V0
        self.imp = int(N/2)
        self.t = -1
        self.w = w
        
        self.Du = np.zeros([N*Q,N*Q])
        self.Dd = np.zeros([N*Q,N*Q])
        self.Gu = np.zeros([N*Q,N*Q]) #pert green function
        self.Gd = np.zeros([N*Q,N*Q])
        self.DOSu = np.zeros([Q, Q])
        self.DOSd = np.zeros([Q, Q])
        self.err = 0.0
        self.tol = tol
        self.gamma = 1e-9j
    
        #Hamiltonian Terms
        self.h1u = np.zeros([N*Q,N*Q])
        self.h1d = np.zeros([N*Q,N*Q])
        
        #Technical Terms
        self.termination = False
        
               
    def Guess_Den(self):
        
        #Average particle density over all positions
        Avg = self.Ns/3
        self.Du = np.diag(np.repeat((1 - Avg), self.N*self.Q))
        self.Dd = np.diag(np.repeat(Avg, self.N*self.Q))
        #Off diagonal terms - supressed
        if np.sum(np.abs(self.Vi)) != 0.0:
            self.Dd += (np.ones([self.N*self.Q,self.N*self.Q]) - np.identity(self.N*self.Q)) * 0.2 * Avg
            self.Du += (np.ones([self.N*self.Q,self.N*self.Q]) - np.identity(self.N*self.Q)) * 0.2 * (1-Avg)
        
    
    def Gen_Pert(self): #!!!ADAPTED - Improve Efficiency
        self.h1u = np.zeros([self.N*self.Q,self.N*self.Q])
        self.h1d = np.zeros([self.N*self.Q,self.N*self.Q])
        frac = self.imp/self.N
        s = int(np.ceil(self.Vi.shape[0]/2)) #We assume square perturbing matrix!
        ss = int(np.floor(self.Vi.shape[0]/2))
        
        #Add Local External Field. i, j - temporal. Periodic Driving.
        if np.sum(np.abs(self.Vi)) != 0.0:
            for i in range(0, self.Q*self.N, self.N):
                for j in range(0, self.Q*self.N, self.N):
                    self.h1u[int(i + self.imp - ss):int(i + self.imp + s), int(j + self.imp - ss):int(j + self.imp + s)] = self.Vi * (int(i+self.N==j) + int(i-self.N==j))
                    self.h1d[int(i + self.imp - ss):int(i + self.imp + s), int(j + self.imp - ss):int(j + self.imp + s)] = self.Vi * (int(i+self.N==j) + int(i-self.N==j))
        
        #Add Static External field
        if self.V0 != 0.0:
            for q in range(self.Q):
                ind = self.imp + q*self.N
                self.h1u[ind, ind] += self.V0
                self.h1d[ind, ind] += self.V0
           
        #Add mean Field
        for i in range(self.Q*self.N):
            for j in range(self.Q*self.N):
                if np.isclose((i/self.N)%1, frac) and np.isclose((j/self.N)%1, frac):
                    self.h1u[i, j] += self.U * self.Dd[i, j]
                    self.h1d[i, j] += self.U * self.Du[i, j]
        return
      
    #!!! Not efficient. High call number
    def G0(self, E, e0=0):
        g0 = 1 / (np.emath.sqrt((E - e0)**2 - 4*(self.t)**2))
        x = (E-e0)/(2*self.t)
        rho1 = -x + np.emath.sqrt(x**2 - 1)
        
        g = lambda l, m: g0 * rho1**(abs(l - m))
        return np.fromfunction(g,  (self.N, self.N))
    
    #ADAPTED.
    def TG0(self, E): 
        tg = []
        for q in range(-(self.Q-1)//2, (self.Q+1)//2):
            tg.append(self.G0(E, q*self.w))
        return block_diag(*tg)
        
    
    def Dyson(self): #ADAPTED
        
        self.Gu = lambda E: self.TG0(E) @ lg.inv(np.identity(self.N*self.Q) - self.h1u @ self.TG0(E))
        self.Gd = lambda E: self.TG0(E) @ lg.inv(np.identity(self.N*self.Q) - self.h1d @ self.TG0(E))
        
              
    #Create DoS of system
    def Gen_DoS(self): #ADAPTED
        self.DOSu = lambda E: abs(-(1.0/np.pi) * np.imag(self.Gu(E + self.gamma)))
        self.DOSd = lambda E: abs(-(1.0/np.pi) * np.imag(self.Gd(E + self.gamma)))
        
        

    def Calc_Error(self, Du2, Dd2): #ADAPTED    
        err_vec = np.zeros(2 * (self.N*self.Q)**2)
        
        err_vec[:(self.N*self.Q)**2] = np.reshape(np.subtract(self.Du, Du2),
                                           (self.N*self.Q)**2)
        err_vec[(self.N*self.Q)**2:] = np.reshape(np.subtract(self.Dd, Dd2),
                                           (self.N*self.Q)**2)
        self.err = lg.norm(err_vec)    
    
    
    
    def Calc_New_Den(self, damping=0.2): 
        
        d = []
        u = []
        ns = np.linspace(2 * self.t, self.Ef, 2001)
        for e in ns:
            #ns = np.linspace(2*self.t, self.Ef, 2001)
            u.append(self.DOSu(e))
            d.append(self.DOSd(e))
        
        u = np.stack(u, axis=0)
        d = np.stack(d, axis=0)
        
        Du2 = np.zeros([self.N*self.Q, self.N*self.Q])
        Dd2 = np.zeros([self.N*self.Q, self.N*self.Q])
        #Need lots of integrals here. Must do one my by one.
        for i in range(self.N*self.Q):
            for j in range(self.N*self.Q):
                Du2[i, j] = np.trapz(np.squeeze(u[:, i, j]), ns)
                Dd2[i, j] = np.trapz(np.squeeze(d[:, i, j]), ns)
        
        self.Calc_Error(Du2, Dd2)

        if self.err < 0.05:
            damping = self.err*damping
        elif self.err < 0.3:
            damping = damping*0.85
            
        self.Du = self.Du * damping + Du2 * (1 - damping)
        self.Dd = self.Dd * damping + Dd2 * (1 - damping)
        
    
    def run_SCF(self, damping=0, verbose=True):
        
        self.err = 1.0
        niter = 0
        
        while self.tol < self.err:

            self.Gen_Pert()
            self.Dyson()
            self.Gen_DoS()
            
            self.Calc_New_Den(damping)
            if niter % 50 == 0 and verbose:
                index = self.imp + (self.Q//2)*self.N
                print('Cycle ', niter, ' Finished. Err:', round(self.err, 6))
                print('Density(q=', self.Q ,'):', np.round(self.Du[index, index], 4),
                      np.round(self.Dd[index, index], 4))
                #print('Pert:', np.round(self.h1u[index, index], 4),
                #      np.round(self.h1d[index, index], 4))
            niter +=1
            if niter >= 1.2e3:
                self.termination = True
                print('Breakpoint Reached. Terminating')
                break
        
        print('Cycle ', niter, ' Finished. Err:', round(self.err, 6))
        print('Density(q=0):', np.round(self.Du[index, index], 4),
              np.round(self.Dd[index, index], 4))
        #print('Pert:', np.round(self.h1u[index, index], 4),
        #      np.round(self.h1d[index, index], 4))
        print('Hartree-Fock SCF analysis finished.', niter, 'cycles.')
        
        
    #Get n(t) from density matrix
    def n_t(self, t, norm):
        
        es = np.fromfunction(lambda m, n: np.exp(1j * (m-n) * self.w * t),
                             [self.Q*self.N, self.Q*self.N])
        print(es)
        
        if norm == True:
            a = (1 / np.sqrt(2*np.pi))
        else:
            a = 1
        
        nu = a*np.sum(np.multiply(self.Du, es))
        nd = a*np.sum(np.multiply(self.Dd, es))
        
        return (nu, nd)


class Plotter:
    
    def __init__(self, model):
        self.model = model
        self.N = model.N
        self.Q = model.Q
        

    
    def P_DOS(self, Es):
        figu, axu = plt.subplots()
        #figd, axd = plt.subplots()
        
        dosu = []
        dosd = []
        #Get DoS for all possible Cases
        for e in Es:
            dosu.append(self.model.DOSu(e))
            dosd.append(self.model.DOSd(e))
        dosu = np.stack(dosu, axis=0)
        dosd = np.stack(dosd, axis=0)
            
        for q in range(self.Q):
            index = q*self.N + self.N//2
            #print(index)
            axu.plot(Es + q*4.5, dosu[:, index, index], label='Q=' + str(q - self.Q//2),
                     alpha=0.9)
            #axd.plot(Es + q*0.3, dosd[:, index, index], label='Q=' + str(q - self.Q//2),
            #         alpha=0.6)
        figu.legend()
        #figd.legend()        
        return        


    def P_Nt(self, ts, ax, ax2=None, label=None, both=False):
        if ax2==None:
            ax2 = ax
        nus = []
        nds = []
        for t in ts:
            nu, nd = self.model.n_t(t)
            nus.append(nu)
            nds.append(nd)
    
        print(nus)
        
        ax.plot(ts, nus, label=label)
        if both:
            ax2.plot(ts, nds, label=label)
        return


