#!/usr/bin/env python

import os, time
from subprocess import Popen, PIPE

isrunning = True

now = time.strftime("%Y%m%d-%H%M%S")
l = open("bluerssi-%s.log" % (now), "w")

while isrunning:
    # p = Popen(["hcitool", "rssi", "68:86:E7:02:1C:05"])
    # print "p", p.communicate()
    output = Popen(["hcitool", "rssi", "68:86:E7:02:1C:05"], stdout=PIPE).communicate()[0]
    rssi = float(output.split(" ")[-1])
    # print "rssi", rssi
    print("%s %f" %(time.strftime("%s"), rssi))
    l.write("%s %f\n" %(time.strftime("%s"), rssi))
    l.flush()
    # a = os.system("hcitool rssi 68:86:E7:02:1C:05")
    time.sleep(1)

l.close()
