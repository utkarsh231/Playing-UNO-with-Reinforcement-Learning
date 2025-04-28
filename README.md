Uno-RL Agent

Uno-RL is a Deep Q-Learning based reinforcement learning agent trained to play the game of Uno from scratch using a custom-built environment and reward system.

The project demonstrates deep reinforcement learning applied to a multi-player, turn-based card game, using TensorFlow/Keras, experience replay, target networks, and Weights & Biases logging.

⸻

Project Overview

	Details
Training time	109.5 hours
Frameworks	TensorFlow, Keras, Wandb, Numpy
RL Algorithm	Deep Q-Network (DQN)
Environment	Custom Uno Game Engine
Logging	Tensorboard + Weights & Biases

The agent was trained over 140k+ steps with experience replay, target network synchronization, and epsilon-greedy exploration.

⸻

Key Components

Custom Uno Environment
	•	Full Uno gameplay logic including:
	•	Regular and special cards (Skip, Reverse, Draw Two, Wild, Wild Draw Four)
	•	Turn-based system with reversing directions
	•	Draw pile management
	•	State representation:
	•	Current top card one-hot encoding
	•	Player hand card counts
	•	Cards to draw information
	•	Action space:
	•	Play a specific card or draw a card

Reward Structure

Original rewards (first run):
	•	Illegal move: -2
	•	Drawing a card: -1
	•	Playing a card: +2
	•	Winning the game: +10

Updated reward structure (planned for second run, based on improvements):
	•	Illegal move: -2
	•	Drawing a card: -0.2
	•	Playing a card: +1
	•	Playing a special card: +2
	•	Having 2 cards left: +2 (Near Win)
	•	Having 1 card left: +5 (Almost Win)
	•	Winning the game: +10

Model and Training
	•	Q-network: MLP (Multi-layer perceptron) trained using MSE loss.
	•	Replay Buffer: 10,000 most recent transitions.
	•	Batch size: 512
	•	Discount factor (gamma): 0.7
	•	Learning rate: 1e-4
	•	Target network updates: Every 20 steps.
	•	Model saves: Every 20,000 steps.

Epsilon-greedy exploration
	•	Initial epsilon: 1.0
	•	Minimum epsilon: 0.05
	•	Decay rate: 0.995

⸻

Results

Cumulative Reward vs Steps

<img src="/mnt/data/cumulative_Reward.jpeg" width="800">


	•	Cumulative reward stays mostly negative (around -6 on average), suggesting that while the agent learns to survive, winning is still rare under initial reward structure.

Epsilon Decay

<img src="/mnt/data/epsilon runtime.jpeg" width="800">


	•	Smooth epsilon decay from 1.0 to about 0.94 over 140k steps, indicating continuous shift from exploration to exploitation.

Game Length

<img src="/mnt/data/game_length.jpeg" width="800">


	•	The majority of games end in 4-5 moves, suggesting the agent either loses quickly or stabilizes fast.

Loss Curve

<img src="/mnt/data/loss.jpeg" width="800">


	•	Loss is generally low but shows spikes around major replay memory flushes or after rare events.

Mean Reward

<img src="/mnt/data/mean_reward.jpeg" width="800">


	•	Mean reward remains consistently negative (~-1) across most training steps in the initial run.

⸻

Future Improvements (In Progress)

🔵 Reward Engineering:
Modified the reward structure to encourage strategic gameplay:
	•	Positive rewards for nearing win conditions (2 or 1 cards left).
	•	Additional reward for playing special cards (Skip, Draw 2, Wild, etc.).

🔵 Updated Training (second run):
	•	A second run was conducted with the improved rewards.
	•	Comparison of mean reward trends between the two runs:

<img src="/mnt/data/updated_mean_rewards.jpeg" width="800">


	•	Early results show higher mean rewards with the new reward structure, suggesting better learning behavior.

⸻

How to Run
	1.	Install dependencies

pip install -r requirements.txt

	2.	Train the agent

python train.py

	3.	Visualize logs on Tensorboard

tensorboard --logdir=logs/

	4.	Visualize full training metrics via Weights & Biases (optional, if wandb enabled).

⸻

Folder Structure

Folder	Purpose
agent.py	Main DQN agent
environment.py	Custom Uno environment
train.py	Training loop
models/	Saved model checkpoints
logs/	TensorBoard logs
wandb/	Weights & Biases tracking (optional)



⸻

Acknowledgements
	•	Reinforcement learning concepts based on Sutton & Barto’s textbook.
	•	TensorFlow/Keras documentation.
	•	Uno card game rules adapted to fit a reinforcement learning context.

⸻

✨ Notes
	•	Training took 109.5 hours over multiple sessions.
	•	Code ensures model saving at consistent intervals to prevent losses after interruptions.
	•	The environment is fully self-contained, no external Uno engine used.
