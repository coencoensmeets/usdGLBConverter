import logging
from pxr import Usd
from robotusd.robot_structure import USDRobot
from robotusd.usd2gltf import USDToGLTFConverter

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
    
    # stage: Usd.Stage = Usd.Stage.Open("Assets/Go2.usd")
    # robot: USDRobot = USDRobot(stage, "Go2Robot")
    # converter = USDToGLTFConverter(robot)
    # converter.export("Output/Go2.glb")
    
    # stage: Usd.Stage = Usd.Stage.Open("Assets/Robots/Unitree/G1/g1.usd")
    # robot: USDRobot = USDRobot(stage, "G1")
    # converter = USDToGLTFConverter(robot)
    # converter.export("Output/G1.glb")
    
    stage: Usd.Stage = Usd.Stage.Open("Assets/Robots/Franka/franka.usd")
    robot: USDRobot = USDRobot(stage, "Franka")
    converter = USDToGLTFConverter(robot)
    converter.export("Output/Franka.glb")
    
    # stage: Usd.Stage = Usd.Stage.Open("Assets/Robots/Festo/FestoCobot/festo_cobot.usd")
    # robot: USDRobot = USDRobot(stage, "Festo")
    # converter = USDToGLTFConverter(robot)
    # converter.export("Output/Festo.glb")