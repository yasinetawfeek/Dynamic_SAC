from pycarmaker import CarMaker, Quantity
import time

# 1 - Open CarMaker with option -cmdport
'''
    For example: on a Windows system with CarMaker 8.0.2 installed on the default
    folder send the command C:\IPG\carmaker\win64-8.0.2\bin\CM.exe -cmdport 16660
'''

# 2 - Start any TesRun

# 3 - Initialize pyCarMaker
IP_ADDRESS = "localhost"
PORT = 16660
cm = CarMaker(IP_ADDRESS, PORT)

# 4 - Connect to CarMaker
cm

# print(dir(CarMaker))

# time.sleep(5)

# cm_x = Quantity("Vehicle.Dynamic.X", Quantity.FLOAT)
# cm_y = Quantity("Vehicle.Dynamic.Y", Quantity.FLOAT)

# cm.DVA_write(cm_x, 500)

# start_time = time.time()
# while time.time() - start_time < 5:
#     cm.send("Set DM.Steer.Ang -2.0")


veh_speed = Quantity("Car.v", Quantity.FLOAT)
veh_speed.data = -1.0

cm.subscribe(veh_speed)

cm.read()
cm.read()


for i in range(10):
    cm.read()  # Update values from the simulation
    print(f"Vehicle speed: {veh_speed.data * 3.6:.2f} km/h")
    time.sleep(1)

cm.send("StopSim")

# 7. Disconnect when done


# Set Vehicle.Dynamic.Y 20.0


# # 5 - Subscribe to vehicle speed
# # Create a Quantity instance for vehicle speed (vehicle speed is a float type variable)
# vehspd = Quantity("Car.v", Quantity.FLOAT)

# # Initialize with negative speed to indicate that value was not read
# vehspd.data = -1.0

# # Subscribe (TCP socket need to be connected)
# cm.subscribe(vehspd)

# # Let's also read the simulation status (simulation status is not a quantity but a command
# # so the command parameter must be set to True)
# sim_status = Quantity("SimStatus", Quantity.INT, True)
# cm.subscribe(sim_status)

# # 6 - Read all subscribed quantities. In this example, vehicle speed and simulation status
# # For some reason, the first two reads will be incomplete and must be ignored
# # You will see 2 log errors like this: [ ERROR]   CarMaker: Wrong read
# cm.read()
# cm.read()
# time.sleep(0.1)
# c = 5
# while(c > 0):
#     c = c - 1
#     # Read data from carmaker
#     cm.read()
#     print()
#     print("Vehicle speed: " + str(vehspd.data * 3.6) + " km/h")
#     print("Simulation status: " + ("Running" if sim_status.data >=
#                                    0 else cm.status_dic.get(sim_status.data)))
#     time.sleep(1)
