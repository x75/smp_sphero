

# Sensorimotor learning with the Sphero robot

05/2014 - 2017, Oswald Berthold

Python code from the Closed-loop sphero behaviour paper <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>.


## Preparation

This is built on top of the following stacks, make sure to have that
installed (tested only on Ubuntu) 

-   python-numpy, python-bluetooth, &#x2026; Use apt-get / distro mechanism
-   ROS base install, use apt-get
-   sphero\_ros driver from <sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup>, install via $ROSDISTRO\_workspace
    catkin\_make. See the ROS wiki how to set up a catkin workspace.
-   get smp\_msgs <sup><a id="fnr.3" class="footref" href="#fn.3">3</a></sup> and install it into your ROS workspace
-   get smp\_base from <https://github.com/x75/smp_base>, then do

    export PYTHONPATH=../smp_base:$PYTHONPATH

This is the launch sequence for the sphero ROS node

    roscore    
    python src/sphero_ros/sphero_node/nodes/sphero.py

\`sphero.py\` accepts arguments such as  sensorimotor loop frequency
and also a BT target address to connect directly to a given device, if
you know it. The experiments are tuned to 20 Hz sampling rate so we
need to do

    python src/sphero_ros/sphero_node/nodes/sphero.py --freq 20

or

    python src/sphero_ros/sphero_node/nodes/sphero.py --freq 20 --target_addr sp:he:ro:sa:dd:re


## Scripts

When roscore and sphero.py are up and running, you can run ROS clients
like e.g. sphero\_joystick.py 

    rosrun joy joy_node

A few of the utils/tests to check the basic communication is working

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">**Utils**</th>
<th scope="col" class="org-left">&#xa0;</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">bluerssi.py</td>
<td class="org-left">Log BT rssi values</td>
</tr>


<tr>
<td class="org-left">sphero\_colors.py</td>
<td class="org-left">Basic color actuation with ros</td>
</tr>


<tr>
<td class="org-left">sphero\_joystick.py</td>
<td class="org-left">Control sphero via ros joy\_node</td>
</tr>


<tr>
<td class="org-left">sphero\_raw.py</td>
<td class="org-left">Minimal test of raw communication protocol</td>
</tr>


<tr>
<td class="org-left">sphero\_simple\_openloop.py</td>
<td class="org-left">Simple open-loop command test</td>
</tr>


<tr>
<td class="org-left">sphero\_test.py</td>
<td class="org-left">Another minimal test</td>
</tr>
</tbody>
</table>


### Learning

When that works, you could try the learning experiments, just to
disentangle possible sources of errors.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">**Experiments**</th>
<th scope="col" class="org-left">&#xa0;</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">sphero\_res\_learner\_1D.py</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero\_res\_learner\_2D\_polar.py</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero\_res\_learner\_2D.py</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero\_res\_learner\_1D\_analyze.py</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero\_data\_recorder.py</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">hk2.py</td>
<td class="org-left">Homeostasis and homeokinesis from lpzrobots,</td>
</tr>


<tr>
<td class="org-left">&#xa0;</td>
<td class="org-left">Der & Martius, 201, Playful machines</td>
</tr>
</tbody>
</table>

    python sphero_res_learner_1D.py --config default.cfg

You can copy the default.cfg and start editing it to play around with
different targets and parameters.

Or try the homeokinesis example and play with self.\*\_gain parameters (in the code)

    python hk2.py --mode hk --loop_time 0.05


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> Berthold and Hafner, 2015, Closed-loop acquisition of behaviour on the Sphero robot, <https://mitpress.mit.edu/sites/default/files/titles/content/ecal2015/ch084.html>

<sup><a id="fn.2" href="#fnr.2">2</a></sup> <https://github.com/x75/sphero_ros>

<sup><a id="fn.3" href="#fnr.3">3</a></sup> <https://github.com/x75/smp_msgs>
