# Persistent-activity

This model is inspired by the article (Compte et al.,2000) and an exercise of neural dynamics textbook. 

This model requires the Brian2 package for neural simulations (https://brian2.readthedocs.io/en/stable/).

This is a leaky integrate and fire model that modelize the prefrontal cortex area and external input connections from other area of the brain.

The network is made of two neuron populations, excitatory and inhibitory, every neurons of both population receives a train of stimulations from the external population with a poisson distribution.


<img width="1000" alt="WorkingMemory_NetworkStructure" src="https://user-images.githubusercontent.com/93595122/160181322-468990e5-7f68-4a38-8191-dbdf2d7f7b9f.png">     Figure 1: The network contains an excitatory population and an inhibitory population. All neurons have three channels variables that modelize the state of the NMDA channels, AMPA channels and GABA channels.


![Picture1](https://user-images.githubusercontent.com/93595122/160186440-1bfb8aa2-b7fa-436d-9140-4f27eeadba3b.png)
 Figure 2: In the delay response task, a visual cue is shown on a screen in one of 8 possible positions, the task consists of remembering the position of the visual cue after a short delay period where the cue is hidden.

The present model mimics a network going through the delay response task (Fig.2). Every excitatory neurons in the network has a prefered cue position and will respond to the visual stimulation only if the cue appear in their prefered position. To modelize this phenomenon a fraction of the excitatory population receives a current stimulation for a defined period of time during the simulation (Istim in Fig.1). This represent the cue presentation period. The fraction of excitatory neurons that receive this stimulation depends of the position of the visual cue.

References:

Neural dynamics textbook exercise on spatial memory: https://neuronaldynamics-exercises.readthedocs.io/en/latest/exercises/spatial-working-memory.html

Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X. J. (2000). Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model. Cerebral Cortex, 10(9), 910-923

