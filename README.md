# pose-to-biped
Person tracking and recreation in MuJoCo simualtor on a robot.

## Dependencies

```[bash]
pip install mujuco
pip install --upgrade mujoco
pip install opencv-python
pip install pin-pink
pip install osqp
pip install mediapipe
```
Be aware that pin-pink works better on Linux, we have not been able to get this running on Windows. This library is only needed for initial kinemtics work. 

You will also need the xml files from <a href="https://github.com/google-deepmind/mujoco_menagerie/tree/main">here </a> or urdf files <a href="https://github.com/unitreerobotics/unitree_ros/tree/master">here</a>. The xml h1 file within the xml repo was edited to include hand joints. Replace the repo once with our edited xml <a href="https://github.com/shepai/pose-to-biped/blob/main/Robots/h1.xml">here</a>

For the kinematics we edited the xml file so the robot would stay welded up right, and we could focus on joint position movement. See /Robots for thsoe files. You will still need to use the meshes from the above repo link. 


## Usage
The package is broken into two sections, pose and simulation. Simulation provides the tools to launch the humanoid robot, and control it. The pose is purely for tracking and transformting points.

### Pose
The pose code makes use of mediapipe and 3D tracking estimation. We track via a video feed and convert it to a local coordinate system to be applied on the robot. 

Example code for calling in this method is seen in <a href="https://github.com/shepai/pose-to-biped/blob/main/pose/test.py">test.py</a>

### Simulator 
The simulator uses the unitree h1 robot and provides the tool for controlling it. We have written code that aligns with the pose from the 3D tracking. This is broken into stages.

#### Kinematics and point alignment 

![Robot demo](https://raw.githubusercontent.com/shepai/pose-to-biped/main/assets/output_record.gif)

#### RL transfer 


