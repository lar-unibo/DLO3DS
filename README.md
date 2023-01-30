## Robot


The Panda from Franka is used as robotic setup. This repository requires to install the [fraka-driver](https://dei-gitlab.dei.unibo.it/lar/franka_driver) plus the following additional ros packages:

```
libfranka
franka-ros
```

## FASTDLO ([https://github.com/lar-unibo/fastdlo](https://github.com/lar-unibo/fastdlo))


Download the [trained models](https://drive.google.com/file/d/1OVcro53E_8oJxRPHqGy619rBNoCD3rzT/view?usp=sharing) and place them in the ```checkpoints``` folder inside ```fastdlo_core```.



## Simulation

BlenderProc is used for rendering simulated images. Install BlenderProc via pip with ```pip install blenderproc``` inside the same virtual environment.

Donwload the [DLO models](https://mega.nz/file/0ZkmGLJT#73O7H61yFNSTuwe2t6lW2Ap2egNryhg5t2yYEo4AgQo) and place them inside ```utilities/blender_rendering/data_models```.

# Running

To execute the algorithm in simulation, run the following commands:

```
roslaunch panda_driver panda_launch_sim.launch
```
to start the simulation of the robot in RViz and

```
roslaunch dlo3ds run_dlo3ds.launch
```
to start the algorithm. Notice that this command must be executed inside the virtual environment.



