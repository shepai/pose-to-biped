# code to record motor position and coordinate space over time
from __init__ import MujocoSimulator
import mujoco
import mujoco.viewer
import numpy as np

import copy
sim = MujocoSimulator("C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml")
viewer=mujoco.viewer.launch_passive(sim.model, sim.data)

#set up simulator
def trial(positions,current_depth=0,max_depth=10):
    # set up trial function
    if max_depth==current_depth: 
        return [],[]
    state_bp=copy.deepcopy(sim.model.get_state(sim.data))#   save data
    #   apply gaussian noise to motors 
    noise=np.random.normal(0,5,positions.shape)
    sim.set_position(positions)
    sim.set_step(3)
    viewer.sync()
    X=[]
    Y=[]
    for i in range():
        temp=positions.copy()
        temp[i]=positions[i]+noise[i] #   move one at a time
        #   reset to previous position
        state = copy.deepcopy(state_bp)
        sim.data.set_state(state) 
        x,y=trial(temp,current_depth+1)
        sim.set_step(3)
        viewer.sync()
        if len(x)>0: #actually returned something
            X.append(x)
            Y.append(y)
        #   gather coordinates of hands and legs
        X.append()
        Y.append()
    
    
    
    
    

#main loop
#generate new position
#run trial
if __name__ == "__main__":
sim = MujocoSimulator(
    "C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml"
)
j=0
for i in range(1000):
    j+=1
    #sim.set_position(sim.initial+(j/10000))
    mujoco.mj_step(sim.model, sim.data)
    viewer.sync()