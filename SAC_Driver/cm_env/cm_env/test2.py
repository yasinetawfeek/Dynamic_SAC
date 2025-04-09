import socket
import time

# Connect to IPGControl
HOST = "127.0.0.1"  # CarMaker runs locally
PORT = 16660  # Default IPGControl port
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

def set_variable(var_name, value):
    """Send a command to set a variable in CarMaker."""
    command = f'set {var_name} {value}\n'
    sock.sendall(command.encode())
    response = sock.recv(1024).decode()
    print(response)  # Check response from CarMaker

# Wait a few seconds to ensure simulation is running
# time.sleep(5)

# Change vehicle position instantly
start_time = time.time()
print(start_time)
while time.time() - start_time < 5:
    set_variable("DM.Steer.Ang", 1.0)  # Move to X = 50 meters
# set_variable("Vehicle.Dynamic.Y", 20.0)  # Move to Y = 20 meters

# Close connection
sock.close()
