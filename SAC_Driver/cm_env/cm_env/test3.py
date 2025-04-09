import sys
sys.path.append("/opt/ipg/carmaker/linux64-13.0.1/Python/python3.10")
import cmapi
import matplotlib
from ASAM.XIL.Implementation.Testbench import TestbenchFactory
from ASAM.XIL.Interfaces.Testbench.MAPort.Enum.MAPortState import MAPortState
from ASAM.XIL.Interfaces.Testbench.Common.Error.TestbenchPortException import TestbenchPortException

MyTestbenchFactory = TestbenchFactory()
MyTestbench = MyTestbenchFactory.CreateVendorSpecificTestBench("IPG", "CarMaker",
"13.0")