Sensorimotor learning with the Sphero robot
===========================================

05/2014 - 2017, Oswald Berthold

Python code from the Closed-loop sphero behaviour paper [1].

Dependencies
------------

This is built on top of the following stacks, make sure to have that installed (tested only on Ubuntu)

-   python-numpy, python-bluetooth, ... Use apt-get / distro mechanism
-   ROS base install, use apt-get
-   sphero\_ros driver from [2], install via $ROSDISTRO\_workspace catkin\_make. See the ROS wiki how to set up a catkin workspace.
-   get smp\_msgs [3] and install it into your ROS workspace
-   get smp\_base from <https://github.com/x75/smp_base>, then do `export PYTHONPATH=../smp_base:$PYTHONPATH`
-   pybluez, the python bluetooth `pip install pybluez`

This is the launch sequence for the sphero ROS node

``` example
roscore    
python src/sphero_ros/sphero_node/nodes/sphero.py 
```

\`sphero.py\` accepts arguments such as sensorimotor loop frequency and also a BT target address to connect directly to a given device, if you know it. The experiments are tuned to 20 Hz sampling rate so we need to do

``` example
python src/sphero_ros/sphero_node/nodes/sphero.py --freq 20
```

or

``` example
python src/sphero_ros/sphero_node/nodes/sphero.py --freq 20 --target_addr sp:he:ro:sa:dd:re
```

Scripts
-------

When roscore and sphero.py are up and running, you can run ROS clients like e.g. sphero\_joystick.py

``` example
rosrun joy joy_node
```

A few of the utils/tests to check the basic communication is working

| **Utils**                   |                                            |
|-----------------------------|--------------------------------------------|
| bluerssi.py                 | Log BT rssi values                         |
| sphero\_colors.py           | Basic color actuation with ros             |
| sphero\_joystick.py         | Control sphero via ros joy\_node           |
| sphero\_raw.py              | Minimal test of raw communication protocol |
| sphero\_simple\_openloop.py | Simple open-loop command test              |
| sphero\_test.py             | Another minimal test                       |

### Learning

When that works, you could try the learning experiments, just to disentangle possible sources of errors.

| **Experiments**                      |                                              |
|--------------------------------------|----------------------------------------------|
| sphero\_res\_learner\_1D.py          |                                              |
| sphero\_res\_learner\_2D\_polar.py   |                                              |
| sphero\_res\_learner\_2D.py          |                                              |
| sphero\_res\_learner\_1D\_analyze.py |                                              |
| sphero\_data\_recorder.py            |                                              |
| hk2.py                               | Homeostasis and homeokinesis from lpzrobots, |
|                                      | Der & Martius, 201, Playful machines         |

``` example
python sphero_res_learner_1D.py --config default.cfg
```

You can copy the default.cfg and start editing it to play around with different targets and parameters.

Or try the homeokinesis example and play with self.\*\_gain parameters (in the code)

``` example
python hk2.py --mode hk --loop_time 0.05
```

Footnotes
=========

[1] Berthold and Hafner, 2015, Closed-loop acquisition of behaviour on the Sphero robot, <https://mitpress.mit.edu/sites/default/files/titles/content/ecal2015/ch084.html>

[2] <https://github.com/x75/sphero_ros>

[3] <https://github.com/x75/smp_msgs>
