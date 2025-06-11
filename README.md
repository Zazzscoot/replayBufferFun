# Augmenting an experience replay buffer with JAX-planner made trajectories for DQN - Small-scale proof of concept using pyRDDLGym
Welcome to a small proof-of-concept for the idea of augmenting an experience replay buffer with "good" trajectories to aid a DQN's exploration. The current proof of concept is built using pyRDDLGym, and only runs on one small domain. Please feel free to check it out!

## Environment Setup
IMPORTANT: Make sure you have Python 3.10 installed on your system.

Required libraries are listed in `requirements.txt`. Use `VirtualEnv` to quickly get these libaries set up for your machine.

### Using `VirtualEnv`
2. Open a terminal/command prompt in the project directory
3. Create a new virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

Note: Make sure to activate the virtual environment every time you work on this project.

## Running the MountainCar Discrete Examples
### Populating the Replay Buffer
After activating the virtual environment, you can start the populating program.
```python
python populator/replay_buffer_populator.py
```
By default, this program will save the replay buffer in the directory: `replay_buffer_data`.

### Running Training
Before we start the training, please make sure that Line `81` of `rl-consumer/run_double_buffer_dqn.py` points to the correct replay buffer file.
```python
python rl-consumer/run_double_buffer_dqn.py
```
By default, this will run training for 200,000 steps, with half & half replay buffers and save the model to the `models` directory.
### Running Tests
You can try testing your model to see how it performs using the script `rl-consumer/test_sb.py`. Before you start testing, make sure that Line `44` has the correct filename for the model.
```python
python rl-consumer/test_double_buffer_dqn.py
```
## Trying Other Domains
The obvious next step is to expand this proof-of-concept to different domains.

### Required edits to the Replay Buffer Populator.
Significant modifications must be made in `populator/replay_buffer_populator.py` to change the domain.
Edit the environment, instance and configuration as necessary on lines `137` to `139`.
Importantly, make sure you define the state on line `141`. In the MountainCar Discrete example, there are only two states, `pos` and `vel`. Notice that its data type and upper/lower bounds are defined. Please be sure to edit line `141` as necessary for whichever domain you want to use.

### Required edits to Training and Testing Scripts
In `rl-consumer/run_double_buffer_dqn.py`, edit the environment and instance as necessary on lines `78` and `79`.

In `rl-consumer/test_double_buffer_dqn.py`, edit the domain and instance as necessary on lines `41` and `42`

Note: Make sure that the domain is consistent between the populator and training/testing script.

