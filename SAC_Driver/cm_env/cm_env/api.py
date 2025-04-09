import sys
sys.path.append("/opt/ipg/carmaker/linux64-13.0.1/Python/python3.10")

import cmapi
import time

# print(dir(cmapi))

class CarMakerControl:
    def __init__(self, project_path=None):
        """
        Initialize CarMaker Control connection
        Args:
            project_path (str): Path to CarMaker project directory
        """
        self.cm = cmapi.CarMaker()
        if project_path:
            self.cm.LoadProject(project_path)
    
    def load_testrun(self, testrun_name):
        """
        Load a specific TestRun
        Args:
            testrun_name (str): Name of the TestRun file
        """
        self.cm.LoadTestRun(testrun_name)
    
    def start_simulation(self):
        """Start the simulation"""
        print(dir(self.cm))
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.cm.StopSim()
    
    def get_quantity(self, quantity_name):
        """
        Get value of a specific quantity
        Args:
            quantity_name (str): Name of the quantity to read
        Returns:
            float: Current value of the quantity
        """
        return self.cm.GetQuantity(quantity_name)
    
    def set_quantity(self, quantity_name, value):
        """
        Set value of a specific quantity
        Args:
            quantity_name (str): Name of the quantity to set
            value (float): Value to set
        """
        self.cm.SetQuantity(quantity_name, value)
    
    def is_running(self):
        """Check if simulation is running"""
        return self.cm.IsRunning()
    
    def wait_for_sim_end(self, timeout=None):
        """
        Wait for simulation to end
        Args:
            timeout (float): Maximum time to wait in seconds
        """
        start_time = time.time()
        while self.is_running():
            if timeout and (time.time() - start_time > timeout):
                raise TimeoutError("Simulation timeout")
            time.sleep(0.1)

cm_control = CarMakerControl()
cm_control.start_simulation()