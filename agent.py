# agent.py
import os
import random
import threading
import collections
import numpy as np
import tensorflow as tf
from keras import models, layers
import wandb
from logger import TensorflowLogger


class UnoAgent:
    # ---- hyper-params you can tweak from train.py if you like --------------
    REPLAY_MEMORY_SIZE     = 10_000
    BATCH_SIZE             = 512
    DISCOUNT_FACTOR        = 0.7
    MODEL_UPDATE_FREQUENCY = 20      # steps between target-net sync
    MODEL_SAVE_FREQUENCY   = 1_000

    def __init__(self, state_size: int, action_count: int, model_path: str | None = None):
        print("Initializing agent …")
        self.initialized   = False
        self.state_size    = state_size
        self.action_count  = action_count
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.995
        self.replay_memory = collections.deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # TensorBoard / wandb
        self.logger       = TensorflowLogger("logs")
        self.model_folder = f"models/{self.logger.timestamp}"
        os.makedirs(self.model_folder, exist_ok=True)

        # build or load networks ------------------------------------------------
        if model_path is None:
            self.model        = self._build_model()
            self.target_model = self._build_model()
            self.target_model.set_weights(self.model.get_weights())
        else:
            self.model        = models.load_model(model_path, compile=False)
            self.target_model = models.load_model(model_path, compile=False)
            # compile after load so we control the optimizer / loss
            self._compile(self.model)
            self._compile(self.target_model)

        # ---- TensorFlow thread lock:  ALL TF ops happen under this ----------
        self._tf_lock = threading.Lock()

        # ---- wandb run -------------------------------------------------------
        wandb.init(
            project="uno-dqn",
            name=f"run-{self.logger.timestamp}",
            config=dict(
                replay_memory_size=self.REPLAY_MEMORY_SIZE,
                batch_size=self.BATCH_SIZE,
                discount_factor=self.DISCOUNT_FACTOR,
                model_update_frequency=self.MODEL_UPDATE_FREQUENCY,
                model_save_frequency=self.MODEL_SAVE_FREQUENCY,
                epsilon_decay=self.epsilon_decay,
                epsilon_min=self.epsilon_min,
            ),
        )

    # -------------------------------------------------------------------------
    # network helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _compile(model: models.Model):
        model.compile(optimizer="adam", loss="mse")  # no accuracy metric for Q-nets

    def _build_model(self) -> models.Model:
        model = models.Sequential(
            [
                layers.Input(shape=(self.state_size,)),
                layers.Dense(128, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.action_count, activation="linear"),
            ]
        )
        self._compile(model)
        return model

    # -------------------------------------------------------------------------
    # replay memory
    # -------------------------------------------------------------------------
    def update_replay_memory(self, transition):
        state, action, reward, next_state, done = transition
        state       = np.asarray(state, dtype=np.float32).flatten()
        next_state  = np.asarray(next_state, dtype=np.float32).flatten()

        if state.shape != (self.state_size,) or next_state.shape != (self.state_size,):
            print(f"[WARN] Skipping malformed state with shape {state.shape}")
            return

        self.replay_memory.append((state, action, reward, next_state, done))

    # -------------------------------------------------------------------------
    # acting
    # -------------------------------------------------------------------------
    def act(self, state: np.ndarray) -> int:
        state = np.asarray(state, dtype=np.float32).flatten()
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_count - 1)

        with self._tf_lock:
            q_values = self.model.predict(state[None, ...], verbose=0)

        return int(np.argmax(q_values[0]))

    # -------------------------------------------------------------------------
    # learning
    # -------------------------------------------------------------------------
    def train(self, step_counter: int):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return

        minibatch  = random.sample(self.replay_memory, self.BATCH_SIZE)
        states     = np.asarray([t[0] for t in minibatch], dtype=np.float32)
        actions    = np.asarray([t[1] for t in minibatch])
        rewards    = np.asarray([t[2] for t in minibatch], dtype=np.float32)
        next_states = np.asarray([t[3] for t in minibatch], dtype=np.float32)
        dones      = np.asarray([t[4] for t in minibatch], dtype=bool)

        # forward passes under a lock
        with self._tf_lock:
            q_values      = self.model.predict(states, verbose=0)
            next_q_values = self.target_model.predict(next_states, verbose=0)

        # manual Bellman update
        for i in range(self.BATCH_SIZE):
            a = actions[i]
            if dones[i]:
                q_values[i, a] = rewards[i]
            else:
                q_values[i, a] = rewards[i] + self.DISCOUNT_FACTOR * np.max(next_q_values[i])

        # supervised step
        with self._tf_lock:
            hist = self.model.fit(states, q_values, batch_size=self.BATCH_SIZE, verbose=0)

        # ---- logging --------------------------------------------------------
        self.logger.scalar("loss", hist.history["loss"][0])
        self.logger.scalar("epsilon", self.epsilon)
        self.logger.flush()

        wandb.log(
            dict(
                loss=hist.history["loss"][0],
                epsilon=self.epsilon,
                step=step_counter,
            )
        )

        # ---- bookkeeping ----------------------------------------------------
        if step_counter % self.MODEL_UPDATE_FREQUENCY == 0:
            self.target_model.set_weights(self.model.get_weights())
            if not self.initialized:
                print("Agent initialized (target net synced)")
                self.initialized = True

        if step_counter % self.MODEL_SAVE_FREQUENCY == 0 and threading.current_thread().name == "Thread-1":
            path = f"{self.model_folder}/model-{step_counter}.keras"
            with self._tf_lock:
                self.model.save(path)
            wandb.save(path)

        # ε-greedy annealing
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)