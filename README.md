05/2014 - 2017, Oswald Berthold <bertolos@informatik.hu-berlin.de>


# Sensorimotor learning experiments with the Sphero robot (smp<sub>sphero</sub>)


## Preparation

This is built on top of the following stacks, make sure to have that
installed (tested only on Ubuntu) 

-   python-numpy, python-bluetooth, &#x2026; Use apt-get / distro mechanism
-   ROS base install, use apt-get
-   sphero<sub>ros</sub> driver from <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>, install via $ROSDISTRO<sub>workspace</sub>
    catkin<sub>make</sub>. See the ROS wiki how to set up a catkin workspace.

This is the launch sequence for talking to sphero with the ros driver:

    roscore    
    python src/sphero_ros/sphero_node/nodes/sphero.py

sphero.py accepts arguments such as update frequency sensorimotor loop
and now also a BT target address to connect directly to a given device

    python src/sphero_ros/sphero_node/nodes/sphero.py --freq 10

    python src/sphero_ros/sphero_node/nodes/sphero.py --freq 20 --target_addr sp:he:ro:sa:dd:re

Optionally, e.g. for sphero<sub>joystick.py</sub>, you need

    rosrun joy joy_node


## Scripts

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Utils</th>
<th scope="col" class="org-left">&#xa0;</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">bluerssi.py</td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero<sub>colors.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero<sub>joystick.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero<sub>raw.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero<sub>simple</sub><sub>openloop.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero<sub>test.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Experiments</td>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">sphero<sub>res</sub><sub>learner</sub><sub>1D</sub><sub>analyze.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero<sub>res</sub><sub>learner</sub><sub>1D.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero<sub>res</sub><sub>learner</sub><sub>2D</sub><sub>polar.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero<sub>res</sub><sub>learner</sub><sub>2D.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>


<tr>
<td class="org-left">sphero<sub>sensorimotor</sub><sub>infth</sub><sub>measures.py</sub></td>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>
</table>


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> <https://github.com/x75/sphero_ros>