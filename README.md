<h1 align="center"><img src="figures/dlo3ds_logo.png"</h1>

<div align="center">
<p> DLO3DS: Deformable Linear Objects 3D Shape Estimation and Tracking from Multiple 2D Views </p>
</div>



# Setting Up

Apart from the standard packages obtained via a Desktop installation of ROS ```(ros-noetic-desktop)```, additional dependencies are:

```
moveit-msgs
moveit-core
moveit-visual-tools
moveit-ros-planning-interface
```

## Robot


The Panda from Franka is used as robotic setup. This repository requires to install the [fraka-driver](https://dei-gitlab.dei.unibo.it/lar/franka_driver/-/tree/panda_mini)(panda_mini branch) plus the following additional ros packages:

```
libfranka
franka-ros
```

## Ariadne+

dependencies:
```
python 3.8
pytorch
cuda 10.1
opencv
scikit-image
python-igraph 0.8.3
```

use conda for creating a virtual environment and name it ```ariadneplus```.

Download the [trained models](https://mega.nz/file/YI90UADT#amRnVdUE4YZcXgO9oBBh9xPRsTA0eP3Py4rSHeU3JS4) and place them inside the ```checkpoints``` folder.

NOTICE: pytorch can be also installed cpu-only and the networks can be executed on the cpu (this is done by defualt if there are no cuda devices available).


## Simulation

BlenderProc is used for rendering simulated images. Install BlenderProc via pip with ```pip install blenderproc``` inside the same ```ariadneplus``` virtual environment.




# Running

To execute the algorithm in simulation, run the following commands:

```
roslaunch panda_driver panda_launch_sim.launch
```
to start the simulation of the robot in RViz and

```
roslaunch dlo3ds dlo3ds.launch
```
to start the algorithm. Notice that this command must be executed inside the virtual environment (example: ```(ariadneplus) alessio@laptop:~/ros/dlo3ds_ws$ roslaunch dlo3ds dlo3ds.launch simulation:=true```) otherwise errors could occur.
)

# IMPORTANT
After setting up the repository and the virtual environment, it is important to change the shebang line in the ```dlo3ds_pipeline.py``` script with the new path of the environment, example: 
```#! /home/alessio/anaconda3/envs/ariadneplus/bin/python```



