# Robot Reverse Replays

Full code to support the robot reverse replay model using the MiRo robot.

To run this code, you must install the MiRo Developer Kit. Instructions for installation can be found at http://labs.consequentialrobotics.com/miro-e/docs/index.php?page=Introduction

Once installed, you must run the Gazebo simulator as per the above instructions. When the simulator is running, you can
 test the robot replay model by running the following scripts:
 
 1. `generate_reward.py` - Generates a reward of value 2 at location (x, y) = (0, 0.6). The location and reward value
  can be changed inside the
  script if
  necessary.
 2. `robo_replay_model.py` - Starts the main hippocampal reverse replay script as described by the model (see the
  paper).
 3. `live_plotting.py` - Produces a realtime plot of the network rates and intrinsic excitabilities.
 4. `miro_controller.py` - This is an example controller script that gets MiRo moving. It causes MiRo to perform a
  random walk, avoids wall collisions, and stops at the reward location for a few seconds whilst reverse replays are
   initiated.