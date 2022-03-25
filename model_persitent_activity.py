import numpy as np
import matplotlib.pyplot as plt
import math
from brian2 import *

start_scope()
defaultclock.dt = 0.05 *ms

# define size of the network
N_e=1024; N_i=256 # number of excitatory and inhibitory neurons
N_extern_poisson=1000 # Size of the external input population (Poisson input)
poisson_firing_rate=1.4 *Hz # Firing rate of the external population


# parameters of conductances
G_inhib2inhib=1.024 * nS # conductance of interneurons on interneurons
G_inhib2excit=1.336 * nS # conductance of interneurons on pyramidals
G_excit2excit=0.381 * nS # conductance of pyramidals on pyramidals
G_excit2inhib=0.292 * nS # conductance of pyramidals on interneurons
    

# parameters excitatory pyramidal cells:
Cm_excit = 0.5 * nF  # membrane capacitance of excitatory neurons
G_leak_excit = 25.0 * nS  # leak conductance
E_leak_excit = -70.0 * mV  # reversal potential
vth_excit = -50.0 * mV  # spike condition
v_reset_excit = -60.0 * mV  # reset voltage after spike
t_abs_refract_excit = 2.0 * ms  # absolute refractory period

# parameters inhibitory interneurons:
Cm_inhib = 0.2 * nF
G_leak_inhib = 20.0 * nS
E_leak_inhib = -70.0 * mV
vth_inhib = -50.0 * mV
v_reset_inhib = -60.0 * mV
t_abs_refract_inhib = 1.0 * ms

# parameters AMPA synapses
E_e = 0.0 * mV
tau_AMPA = .9 * 2.0 * ms

# parameters GABA synapses
E_GABA = -70.0 * mV
tau_GABA = 10.0 * ms

# parameters NMDA synapses
E_e = 0.0 * mV
tau_NMDA_s = 1 * 100.0 * ms  # orig: 100
tau_NMDA_x = 1.5 * 2.0 * ms
alpha_NMDA = .5 * kHz

# projections from the external population
G_extern2inhib = 2.38 * nS
G_extern2excit = 3.1 * nS

# chose stimulus position
stimulus_pos=4

# define subpopulation affected by sensory simulus

j=int(N_e/8)
sensory_stim=[[0,j]]
for a in range(1,8):
    sensory_stim.append([a*j,(a+1)*j-1]) # group the neurons in 8 equal groups depending of thri id#
    
# define sensory stimulus 
I_stim_def=[]
idx=0
while idx<N_e:
    if idx>sensory_stim[stimulus_pos][0] and idx<sensory_stim[stimulus_pos][1]:
        I_stim_def.append([0,.1,0,0])
    else:
        I_stim_def.append([0,0,0,0])
    idx+=1
I_stim=TimedArray(np.vstack(I_stim_def).T, dt=200*ms)

# define the inhibitory population
eqs_inhib = """
    dv/dt = (- G_leak_inhib * (v-E_leak_inhib)- G_extern2inhib * s_AMPA * (v-E_e)- G_inhib2inhib * s_GABA * (v-E_GABA)- G_excit2inhib * s_NMDA  * (v-E_e)/(1.0+1.0*exp(-0.062*1e3*v/volt)/3.57))/Cm_inhib : volt (unless refractory)
    ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
    ds_GABA/dt = -s_GABA/tau_GABA : 1
    ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
    dx/dt = -x/tau_NMDA_x : 1
"""

inhib_pop = NeuronGroup(N_i, model=eqs_inhib,threshold="v>vth_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,method="rk2")

# initialize with random voltages:
inhib_pop.v = np.random.uniform(v_reset_inhib / mV, high=vth_inhib / mV,size=N_i) * mV

# set the connections: inhib2inhib
syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * ms)
syn_inhib2inhib.connect(condition="i!=j", p=1.0)

# set the connections: extern2inhib
input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",N=N_extern_poisson, rate=poisson_firing_rate, weight=1)


# define the excitatory population:
eqs_excit = """
    dv/dt = (- G_leak_excit * (v-E_leak_excit)- G_extern2excit * s_AMPA * (v-E_e)- G_inhib2excit * s_GABA * (v-E_GABA)- G_excit2excit * s_NMDA * (v-E_e)/(1.0+1.0*exp(-0.062*1e3*v/volt)/3.57)+I_stim(t,i)*namp)/Cm_excit : volt (unless refractory)
    ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
    ds_GABA/dt = -s_GABA/tau_GABA : 1
    ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1 
    dx/dt = -x/tau_NMDA_x : 1
"""

excit_pop = NeuronGroup(N_e, model=eqs_excit,threshold="v>vth_excit", reset="v=v_reset_excit; x+=1.0",refractory=t_abs_refract_excit, method="rk2")

# initialize with random voltages:
excit_pop.v = np.random.uniform(v_reset_excit / mV, high=vth_excit / mV,size=N_e) * mV

# set the connections from extern
input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",N=N_extern_poisson, rate=poisson_firing_rate, weight=1)


# set the connections: inhibitory to excitatory
syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
syn_inhib2excit.connect(p=1.0)

# set the connections: excitatory to inhibitory NMDA connections
syn_excit2inhib = Synapses(excit_pop, inhib_pop,on_pre="x += 0.5 ; s_AMPA +=0.1 ") 
syn_excit2inhib.connect(p=1.0)

# set the connections: excitatory to inhibitory NMDA connections
syn_excit2excit = Synapses(excit_pop, excit_pop,on_pre="x+= 2; s_AMPA +=2 ") 
syn_excit2excit.connect(p=1.0,condition="abs(i-j)<9") 

#create the monitors
M = SpikeMonitor( excit_pop )
S = StateMonitor( excit_pop, ('v','s_NMDA','s_AMPA','s_GABA'), record=True )

#run simulation
run(800*ms,report='text')


# ratio of firing between prefered and unprefered neurons during the stimulation
prefered=0
unprefered=0
b=0
for a in M.t/ms:
    b+=1
    if 200<a<400 and sensory_stim[stimulus_pos][0]-100<M.i[b]<sensory_stim[stimulus_pos][1]+100:
        prefered+=1
    elif 200<a<400 and M.i[b] not in range(sensory_stim[stimulus_pos][0],sensory_stim[stimulus_pos][1]):
        unprefered+=1
during_I=[prefered,unprefered]

# ratio of firing between prefered and unprefered neurons after stimulation
prefered=0
unprefered=0
b=0
for a in M.t/ms:
    if a>400 and sensory_stim[stimulus_pos][0]-100<M.i[b]<sensory_stim[stimulus_pos][1]+100:
        prefered+=1
    elif a>400 and M.i[b] not in range(sensory_stim[stimulus_pos][0],sensory_stim[stimulus_pos][1]):
        unprefered+=1
    b+=1
if unprefered==0:
    unprefered=1
after_I=[prefered,unprefered]



#plot excitatory spiking activity
figure(figsize=(15,5))
xlabel( 'time (ms)'); ylabel( 'cell id' )
plot( M.t/ms, M.i,'.k' )
show()

# plot membrane voltage of one neuron
figure(figsize=(15,5))
xlabel( 'time (ms)' ); ylabel( 'neuron id#' )
plot( S.t/ms, S.v[600] )
show()
print("During stimulation: \n The prefered neurons spike {prefered} times\n Unprefered neurons spike {unprefered}".format(prefered=during_I[0],unprefered=during_I[1]))
print("After the stimulation:\n The prefered neurons spike {prefered} times \n Unprefered neurons spike {unprefered}".format(prefered=after_I[0],unprefered=after_I[1]))
