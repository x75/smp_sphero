import rospy

from std_msgs.msg import Float32, Float32MultiArray, ColorRGBA

import numpy as np

def main():
    rospy.init_node("sphero_color")
    pub = {}
    pub["color"] = rospy.Publisher("/set_color", ColorRGBA, queue_size=2)
    rate = rospy.Rate(1)
    color = ColorRGBA()
    color.r = 1.

    while not rospy.is_shutdown():
        print("loop")
        # r, g, b = np.random.uniform(0, 1, (3,))
        r, g, b = (0.04, 0, 1)
        color.r = r
        color.g = g
        color.b = b
        pub["color"].publish(color)
        rate.sleep()

if __name__ == "__main__":
    main()
