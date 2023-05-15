import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#%% --- Setting up V_eff matrix ---

class V_Effective:
    
    def __init__(self, INTS=1):
        self.N=INTS
        self.MAXCALL= 20
        
        #possible Impurities
        self.V0 = lambda V1: np.identity(INTS) * V1/2
        
        self.V = lambda V1: np.array([[0, (V1*1j)],[(V1*1j).conjugate(),0]]) * 1/2

        self.V2 = lambda V1: np.array([[0, (-V1*1j), 0],
                        [(-V1*1j).conjugate(), 0, (V1*1j).conjugate()],
                        [0, (V1*1j), 0]]) * 1/2
        
        self.V02 = lambda V1: np.array([[0, -(V1*1j), 0],
                        [-(V1*1j).conjugate(), V1*2, (V1*1j).conjugate()],
                        [0, (V1*1j), 0]]) * 1/2
        

    @staticmethod
    def g_lm(E, T, l, m):
        g0 = 1/(E * np.emath.sqrt(1 - (4*T**2)/(E**2)))
        x = E/(2*T)
        rho1 = -x + np.emath.sqrt(x**2 - 1)
        return g0 * rho1**(abs(l - m))
      
      
    def G_0(self, E, T):
        el = lambda l, m: self.g_lm(E, T, l, m)
        return np.fromfunction(el,  (self.N, self.N))
    
    
    def Vu(self, E, T, V, C, n=0):
        if n >= self.MAXCALL:
            return np.identity(self.N, dtype=np.float64)
        else:
            dn = lg.inv(C(E+1, T)) - (V @ self.Vu(E+1, T, V, C, n+1) @ V)
            return lg.inv(dn)
    
    
    def Vd(self, E, T, V, C, n=0):
        if n >= self.MAXCALL:
            return np.identity(self.N , dtype=np.float64)
        else:
            dn = lg.inv(C(E-1, T)) - (V @ self.Vd(E-1, T, V, C, n+1) @ V)
            return lg.inv(dn)
    
    #Change Name
    def V_eff(self, E, T, V, C):
        up = V @ self.Vu(E, T, V, C) @ V
        down = V @ self.Vd(E, T, V, C) @ V
        return up + down


#%% --- Computation of G, T and DoS, and t^2

class Impurity(V_Effective):
    
    def __init__(self, INTS):
        #!!!Need to choose impurity
        V_Effective.__init__(self, INTS)
        
        if INTS == 1:
            Vc = self.V0
        elif INTS == 2:
            Vc = self.V1
        elif INTS == 3:
            Vc = self.V2
        else:
            print('Unable to deal with dimensionality')
            return
        
        self.V_e = lambda E, T, V: self.V_eff(E, T, Vc(V), self.G_0)        


    def G_D(self, E, T, V1, V_e, G_s=None):
        if G_s is None:
            G_s = self.G_0
        else:
            print('Changing G_s!')
        #    self.G_0 = G_s
            
        dn = lg.inv(np.identity(self.N) - V_e(E, T, V1) @ G_s(E,T))
        return G_s(E, T) @ dn
    
    
    def TT(self, E, T, V1, V_e):
        #t-matrix
        return V_e(E, T, V1) @ self.G_D(E, T, V1, V_e) @ lg.inv(self.G_0(E, T))
    
    
    def Tf(self, E, T, V1, V_e):
        f = 0
        k = np.arccos(-E/(2*T))
        for i in range(self.N):
            for j in range(self.N):
                f += self.TT(E, T, V1, V_e)[i,j] * np.exp(1j*k*(i-j))
        return f                
                
                
    def DoS(self, arr, T, V_e, gamma=1e-11):
        V1, E = arr
        return abs(- (1/np.pi) * np.imag(self.G_D((E + gamma*1j), T, V1, V_e)))
    
    
    #!!!Can I remove this?
    def PDoS(E, T, V1, V_e, n, gamma=1e-11):
        return (-1/np.pi) * np.imag(5)
    
    
    #Transmission
    def Ti(self, arr, T, V_e, g=1e-11):
        V1, E = arr
        t = 1 + self.G_0(E + g*1j, T)[0,0] * self.Tf(E + g*1j, T, V1, V_e)
        return np.abs(t)**2


#%% --- Ranger ---


class Ranger(Impurity):
    
    def __init__(self, Trng, Vrng, Erng, INTS, DOS=True, TRNS=True, manual=False):
        Impurity.__init__(self, INTS)
        
        if (type(Trng) is int) or (type(Trng) is float):
            print('Using Constant T Procedure')
            self.T = Trng
            if manual:
                self.Vs = np.array(Vrng) * Trng
            else: 
                self.Vs = np.linspace(Vrng[0], Vrng[1], Vrng[2]) * Trng
            self.Es = np.linspace(Erng[0], Erng[1], Erng[2]) * self.T
            
            #Run over values
            V_mat = np.vstack([self.Vs]*len(self.Es))
            E_mat = np.vstack([self.Es]*len(self.Vs)).T
            Vals = np.stack((V_mat, E_mat))
            
            if DOS:
                f = lambda arr: self.DoS([arr[0], arr[1]], self.T, self.V_e)
                self.dos = np.squeeze(np.apply_along_axis(f, 0, Vals))#[0,0,:,:]
                if self.dos.ndim == 4:
                    self.dos = self.dos[0,0,:,:]
            if TRNS:
                f = lambda arr: self.Ti([arr[0], arr[1]], self.T, self.V_e)
                self.trns = np.squeeze(np.apply_along_axis(f, 0, Vals))
        elif type(Trng) is list:
            print('Using Variable T procedure')
            if manual:
                self.T = np.array(Trng)
            else:
                self.T = np.linspace(Trng[0], Trng[1], Trng[2])
            self.Vs = np.linspace(Vrng[0], Vrng[1], Vrng[2])
            VTs = []
            for t in self.T:
                VTs.append(self.Vs*t)
                
            self.Es = Erng
            try:
                if type(Erng) is list : raise Exception()
            except:
                print('Invalid Es. Currently only one value ok')
       
            V_mat = np.vstack(VTs)
            T_mat = np.vstack([self.T]*len(self.Vs)).T
            Vals = np.stack((V_mat, T_mat))
            
            f = lambda arr: self.Ti([arr[0], self.Es*arr[1]], arr[1], self.V_e)
            if TRNS:
                self.trns = np.squeeze(np.apply_along_axis(f, 0, Vals))
       
        
        
    def find_resonances(self, Vrng, height=-1):
        
        #Create Masks over relevant range
        mask = (self.Vs > Vrng[0]) & (self.Vs < Vrng[1])
        m_mask = np.vstack([mask]*len(self.Vs))
        
        #Apply peak finder for relevant runs. Only Transmission.
        self.peaks = dict()
        self.troughs = dict()
        if (type(self.T) is int) or (type(self.T) is float):
            print('Using E as variance Variable')
            for i, e in enumerate(self.Es):
                pks, _ = find_peaks(self.trns[i, mask])
                self.peaks[e] = self.Vs[pks]
        
        else:
            print('Using T as Variable')
            for i, t in enumerate(self.T):
                pks, _ = find_peaks(self.trns[i, mask])
                trs, _ = find_peaks(-self.trns[i, mask], height=height)
                #print(pks)
                self.peaks[t] = self.Vs[mask][pks]
                self.troughs[t] = self.Vs[mask][trs]
                print(self.troughs[t])
            self.mask = mask
    

#%% --- Plotting ---

class Plotter:
    
    def __init__(self, model):
        self.model = model
        
        self.palette = ['#332288', '#117733', '#44AA99', '#88CCEE',
                        '#DDCC77', '#CC6677', '#AA4499', '#882255'] 
        
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.propsl = dict(fancybox=True)
        return
    
    #Only takes in G_00, not any other location.
    def DoS_plt(self, axes, fig, savepath=None, tx = r'$V_{0,1}=$', lw=1.5,
                indiv=False, supy=True):

        if indiv == False:
            print('Subplot Mode')
            if len(axes.flat) != len(self.model.dos[0,:]):
                print('Invalid Axis number. Aborting')
                return
            if type(axes) is np.ndarray:
                for i, ax in enumerate(axes.flat): 
                    ax.plot((self.model.Es)/(self.model.T), self.model.dos[:,i], 
                            c=self.palette[0], lw=lw)
                    
                    ax.text(0.05, 0.95, tx + str(round(self.model.Vs[i]/self.model.T,2)) + r'$J$',
                            transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', bbox=self.props)
                    if i%2==1: ax.yaxis.tick_right()
                    ax.set_ylim(bottom=0)
            else:
                axes.plot((self.model.E)/(self.moel.T), self.model.dos,
                        c=self.palette[0])
            
            
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.supxlabel(r'Energy [E/J]', fontsize=14)
            if supy:
                fig.supylabel(r'$\rho_0$', fontsize=14)
            #fig.suptitle(r'DoS of Floquet Impurity, $T=2\hbar\omega$')
            #txt=r'DoS of Floquet Impurity, $T=2\hbar\omega$'
            #txt=r'DoS of Local Optical Field, $T=2\hbar\omega$'
            if savepath != None:
                fig.savefig(savepath, dpi=1200, bbox_inches='tight')
                print('Plot saved.\n', savepath)
        elif indiv:
            print('Individual Mode')
            if len(axes) != len(self.model.dos[0,:]):
                print('Invalid Axis number. Aborting')
                return
            for i, ax in enumerate(axes): 
                ax.plot((self.model.Es)/(self.model.T), self.model.dos[:,i], 
                        c=self.palette[0], lw=lw)
                
                ax.text(0.05, 0.95, tx + str(round(self.model.Vs[i]/self.model.T,2)) + r'$J$',
                        transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=self.props)
                if i%2==1: ax.yaxis.tick_right()
            
                ax.set_ylim(bottom=0)
                ax.set_xlabel(r'E/J', fontsize=14)
                ax.set_ylabel(r'$\rho_0$', fontsize=14)
            
        return
        
        
    def T_pltV(self, ax, fig, savepath=None, tx='V_{0}', legend=True):
        
        for i, t in enumerate(self.model.T):
            lbl=r'$\omega/J=$'+str(round(1/t, 2))
            ax.plot(self.model.Vs, self.model.trns[i,:], label= lbl,
                    c=self.palette[(2*i+2)%8])

        ax.set_ylabel(r'$|t|^2$', fontsize=14) # [arbitrary units]
        ax.set_xlabel(r'$' + tx + r'/J$', fontsize=14)
        #ax.set_title(r'Transmission through Floquet Hopping, $\epsilon=-0.5$')
        if legend:
            fig.legend(borderaxespad=3, **self.propsl)
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(0, self.model.Vs[-1])
        #ax.set_yscale('log')
        if savepath != None:
            fig.savefig(savepath, dpi=1200, bbox_inches='tight')
            print('Plot saved.\n', savepath)
        
        return


    def T_pltE(self, ax, fig, savepath=None, tx='V_{0}'):

        for i, v in enumerate(self.model.Vs):
            ax.plot(self.model.Es/self.model.T, self.model.trns[:,i],
                    label= r'$\frac{' + tx + r'}{J}=$'+str(v/self.model.T),
                    c=self.palette[i])
            
        ax.set_ylabel(r'$|t|^2$', fontsize=14)
        ax.set_xlabel(r'$Incoming Particle Energy [E/J]$', fontsize=14)
        #ax.set_title(r'Transmission through a local Optical Field, $T=2\hbar\omega$')
        fig.legend(bbox_to_anchor=(0.5, 0.98), loc="center", **self.propsl,
                   ncol=len(self.model.Vs)//2)
        ax.vlines([-1.5, -1, -0.5], ymin=0, ymax=1, 
                  color='k', linestyle='--', lw=0.9, alpha=0.7)
        #txt3=r'Transmission through a local Optical Field, $T=2\hbar\omega$'
        #txt4=r'Transmission through a Floquet Barrier, $T=2\hbar\omega$'
        if savepath != None:
            fig.savefig(savepath, dpi=1200, bbox_inches='tight')
            print('Plot saved.\n', savepath)
        return

            
    
    def T_plt_eff(self, ax, fig, var = None):
        if var is None:
            try:
                var = self.model.eff
            except:
                print('Now effective potential calculated')
                
        for i, t in enumerate(self.model.T):
            ax.plot(self.model.Vs, var[i,:], label=r'$J=$'+str(t))#, c=palette[2*i%8])
            #fig.legend(borderaxespad=3, **self.propsl)
        ax.set_yscale('log')
       
        
    def plt_res(self, ax, fig):
        
        Ts = []
        Vs = np.array([])
        for t in self.model.troughs:
          Ts.extend([t] * len(self.model.troughs[t]))
          Vs = np.concatenate((Vs, self.model.troughs[t]))
         
        ax.scatter([1/i for i in Ts], Vs, marker='x', color='k')        
        ax.set_ylabel(r'Driving Potential [$V_1/J$]')
        ax.set_xlabel(r'Driving Frequency [$\omega / J$]')
        
       
    def T_pltEV(self, ax):
        T = 2

        Es = np.linspace(-2*T, 0, pts)
        Vs = np.linspace(0*T, 10*T, pts)
        Ts_VE = np.zeros([pts, pts])

        for v in range(pts):
            for e in range(pts):
                Ts_VE[v, e] = Ti(Es[e], T, Vs[v], V_e)
            
        fig, ax = plt.subplots(dpi=600)
        C = ax.contourf(Es/T, Vs/T, Ts_VE, 60) 
        cbar = fig.colorbar(C)
        ax.set_ylabel(r'$V_0/T$', fontsize=14)
        ax.set_xlabel(r'$E/T$', fontsize=14)
        cbar.ax.set_ylabel(r'$|t|^2$', fontsize=14)
        cbar.ax.locator_params(nbins=11)
        #ax.set_title(r'Transmission through Floquet ')
        #cbar.set_clim(0, 1.0)
        #plt.clim(0, 1.0) 
        #ax.vlines([-1.5, -1, -0.5], 0.1, 10, color='crimson',
        #          linestyle='--', linewidth=0.8)
        plt.vlines
        
    #Calculate Const(E) Variable (V, T/w), [2D] ---------------------------------


    
#%% --- Model Run ---

#Model = Ranger([0.5, 2, 20], [0.001, 8.0, 100], -0.5, 3)

#%% Peak testing ---

#Model.find_resonances([2.5, 6])

#fig2, axes2 = plt.subplots(dpi=600, figsize=[12,6])
#Plot.plt_res(axes2, fig2)





#%% --- BIN ---
