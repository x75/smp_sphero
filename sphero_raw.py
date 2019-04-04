#!/usr/bin/python
"""sphero raw communication test
"""
# imports
import bluetooth
import struct
import time
import sys

# sphero_driver
from sphero_driver import sphero_driver

# create sphero and connect
sphero = sphero_driver.Sphero()
sphero.connect()
sphero.set_raw_data_strm(40, 1 , 0, False)

# send a few commands to LED, disconnect and exit
sphero.start()
time.sleep(2)
sphero.get_version(True)
time.sleep(2)
sphero.set_rgb_led(255,0,0,0,False)
time.sleep(1)
sphero.set_rgb_led(0,255,0,0,False)
time.sleep(1)
sphero.set_rgb_led(0,0,255,0,False)
time.sleep(3)
sphero.join()
sphero.disconnect()
sys.exit(1)
