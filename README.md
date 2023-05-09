
# Deformable Linear Objects 3D Shape Estimation and Tracking From Multiple 2D Views

IEEE Robotics and Automation Letters (RA-L)

### Abstract
This paper presents DLO3DS , an approach for the 3D shapes estimation and tracking of Deformable Linear Objects (DLOs) such as cables, wires or plastic hoses, using a cheap and compact 2D vision sensor mounted on the robot end-effector. DLO3DS can be applied in all those scenarios in which the perception and manipulation of DLO-like structures are needed, such as in the case of switchgear cabling, wiring harness manufacturing and assembly in the automotive and aerospace industries, or production of hoses for medical applications. The developed procedure is based on a pipeline that first processes the images coming from the 2D camera extracting key topological points along the DLOs. These points are then used to model each DLO with a B-spline curve. Finally, the set of splines obtained from all the images is matched by exploiting a multi-view stereo-based algorithm. DLO3DS is validated both on a real scenario and on simulated data obtained by exploiting a rendering engine for photo-realistic images. In this way, reliable ground-truth data are retrieved and utilized for assessing the estimation error achievable by DLO3DS , which on the employed test set is characterized by a mean reconstruction error of 0.82 mm.

Link IEEE Xplore (open access): [https://ieeexplore.ieee.org/document/10120758](https://ieeexplore.ieee.org/document/10120758)



# How to

### Robot
The Panda from Franka is used as robotic setup. This repository requires to install the [fraka-driver](https://dei-gitlab.dei.unibo.it/lar/franka_driver) plus the following additional ros packages:

```
libfranka
franka-ros
```
### FASTDLO ([https://github.com/lar-unibo/fastdlo](https://github.com/lar-unibo/fastdlo))
Download the [trained models](https://drive.google.com/file/d/1OVcro53E_8oJxRPHqGy619rBNoCD3rzT/view?usp=sharing) and place them in the ```checkpoints``` folder inside ```fastdlo_core```.



### Simulation

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



