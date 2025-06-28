import logging
from pxr import Usd
from robotusd.robot_structure import USDRobot
from robotusd.usd2gltf import USDToGLTFConverter
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set logger to INFO level (no DEBUG messages)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # Enable detailed debug logging to see transform extraction
    # logging.getLogger().setLevel(logging.DEBUG)
    # logger.setLevel(logging.DEBUG)
    
    # usd_file = "Assets/Robots/Unitree/Go2/go2.usd"
    usd_file = "Assets/Robots/Franka/franka.usd" 
    # usd_file = "Assets/Robots/Festo/FestoCobot/festo_cobot.usd"  
    if not os.path.exists(usd_file):
        logger.error(f"USD file {usd_file} does not exist.")
        
    robot_name = usd_file.split('/')[-1].replace('.usd', '')
    robot: USDRobot = USDRobot(Usd.Stage.Open(usd_file), robot_name)
    for joint in robot.joints.values():
        logger.info(f"Joint name: {joint}")
    # converter = USDToGLTFConverter(robot)
    # converter.export(f"Output/{robot_name}.glb")
    