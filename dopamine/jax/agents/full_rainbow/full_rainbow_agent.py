# coding=utf-8
# Copyright 2021 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compact implementation of the full Rainbow agent in JAX.

Specifically, we implement the following components from Rainbow:

    * n-step updates
    * prioritized replay
    * distributional RL
    * double_dqn
    * noisy
    * dueling

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""

import functools

from absl import logging
from dopamine.jax import losses
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rainbow import rainbow_agent
from dopamine.metrics import statistics_instance
from dopamine.replay_memory import prioritized_replay_buffer
import gin
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as onp
import optax
import tensorflow as tf

from flax import linen as nn
import os

from numpy.random import zipf
from collections import deque, namedtuple
from enum import Enum

import pickle


class Strategy(Enum):
    BOLTZMANN = "boltzmann"
    EZ_GREEDY = "ez_greedy"
    E_GREEDY = "e_greedy"


@gin.configurable
def zero_epsilon(
        unused_decay_period, unused_step, unused_warmup_steps, unused_epsilon
):
    return 0.0


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11, 13, 16))
def select_action(
        network_def,
        params,
        state,
        rng,
        num_actions,
        eval_mode,
        epsilon_eval,
        epsilon_train,
        epsilon_decay_period,
        training_steps,
        min_replay_history,
        epsilon_fn,
        support,
        exploration_strategy,
        exploration_steps_left,
        last_action,
        use_trigger,
        exploration_signal,
        explore_duration,
):
    epsilon = jnp.where(
        eval_mode,
        epsilon_eval,
        epsilon_fn(
            epsilon_decay_period,
            training_steps,
            min_replay_history,
            epsilon_train,
        ),
    )

    rng, *rands = jrandom.split(rng, num=5)
    p = jrandom.uniform(rands[0])

    net_out = network_def.apply(
        params, state, key=rands[1], eval_mode=eval_mode, support=support
    )

    best_action = jnp.argmax(net_out.q_values)
    random_action = jrandom.randint(rands[2], (), 0, num_actions)

    # Exploration decision based on VPD or epsilon
    is_exploratory_decision = (
        exploration_signal if use_trigger and not eval_mode else p <= epsilon
    )

    if exploration_strategy == Strategy.BOLTZMANN:
        softmax_values = nn.softmax(jnp.divide(net_out.q_values, epsilon))
        softmax_pick = jrandom.choice(
            rands[3], net_out.q_values, p=softmax_values
        )
        action = jnp.argmax(net_out.q_values == softmax_pick)
        is_exploratory_decision = jnp.not_equal(action, best_action)

    elif exploration_strategy == Strategy.EZ_GREEDY:
        is_zero = jnp.equal(exploration_steps_left, 0)
        action = jnp.where(
            is_zero,
            jnp.where(is_exploratory_decision, random_action, best_action),
            last_action,
        )
        sampled_duration = jnp.where(
            is_exploratory_decision, explore_duration, 0
        )
        exploration_steps_left = jnp.where(
            is_zero, sampled_duration, exploration_steps_left - 1
        )
        is_exploratory_decision = jnp.logical_or(
            is_exploratory_decision, ~is_zero
        )

    elif exploration_strategy == Strategy.E_GREEDY:
        action = jnp.where(is_exploratory_decision, random_action, best_action)

    else:
        raise Exception(
            "Unknown exploration strategy: " + exploration_strategy.value
        )

    return (
        rng,
        exploration_steps_left,
        action,
        net_out.value,
        is_exploratory_decision,
        epsilon
    )


@functools.partial(jax.vmap, in_axes=(None, 0, None))
def get_logits(model, states, rng):
    return model(states, key=rng).logits


@functools.partial(jax.vmap, in_axes=(None, 0, None))
def get_q_values(model, states, rng):
    return model(states, key=rng).q_values


@functools.partial(jax.jit, static_argnums=(0, 3, 12, 13, 14))
def train(
        network_def,
        online_params,
        target_params,
        optimizer,
        optimizer_state,
        states,
        actions,
        next_states,
        rewards,
        terminals,
        loss_weights,
        support,
        cumulative_gamma,
        double_dqn,
        distributional,
        rng,
):
    """Run a training step."""

    # Split the current rng into 2 for updating the rng after this call
    rng, rng1, rng2 = jrandom.split(rng, num=3)

    def q_online(state, key):
        return network_def.apply(online_params, state, key=key, support=support)

    def q_target(state, key):
        return network_def.apply(target_params, state, key=key, support=support)

    def loss_fn(params, target, loss_multipliers):
        """Computes the distributional loss for C51 or huber loss for DQN."""

        def q_online(state, key):
            return network_def.apply(params, state, key=key, support=support)

        if distributional:
            logits = get_logits(q_online, states, rng)
            logits = jnp.squeeze(logits)
            # Fetch the logits for its selected action. We use vmap to perform this
            # indexing across the batch.
            chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
            loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
                target, chosen_action_logits
            )
        else:
            q_values = get_q_values(q_online, states, rng)
            q_values = jnp.squeeze(q_values)
            replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
            loss = jax.vmap(losses.huber_loss)(target, replay_chosen_q)

        mean_loss = jnp.mean(loss_multipliers * loss)
        return mean_loss, loss

    # Use the weighted mean loss for gradient computation.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    target = target_output(
        q_online,
        q_target,
        next_states,
        rewards,
        terminals,
        support,
        cumulative_gamma,
        double_dqn,
        distributional,
        rng1,
    )

    # Get the unweighted loss without taking its mean for updating priorities.
    (mean_loss, loss), grad = grad_fn(online_params, target, loss_weights)
    updates, optimizer_state = optimizer.update(
        grad, optimizer_state, params=online_params
    )
    online_params = optax.apply_updates(online_params, updates)
    return optimizer_state, online_params, loss, mean_loss, rng2


@functools.partial(
    jax.vmap, in_axes=(None, None, 0, 0, 0, None, None, None, None, None)
)
def target_output(
        model,
        target_network,
        next_states,
        rewards,
        terminals,
        support,
        cumulative_gamma,
        double_dqn,
        distributional,
        rng,
):
    """Builds the C51 target distribution or DQN target Q-values."""

    is_terminal_multiplier = 1.0 - terminals.astype(jnp.float32)
    # Incorporate terminal state to discount factor.
    gamma_with_terminal = cumulative_gamma * is_terminal_multiplier

    target_network_dist = target_network(next_states, key=rng)
    if double_dqn:
        # Use the current network for the action selection
        next_state_target_outputs = model(next_states, key=rng)
    else:
        next_state_target_outputs = target_network_dist
    # Action selection using Q-values for next-state
    q_values = jnp.squeeze(next_state_target_outputs.q_values)
    next_qt_argmax = jnp.argmax(q_values)

    if distributional:
        # Compute the target Q-value distribution
        probabilities = jnp.squeeze(target_network_dist.probabilities)
        next_probabilities = probabilities[next_qt_argmax]
        target_support = rewards + gamma_with_terminal * support
        target = rainbow_agent.project_distribution(
            target_support, next_probabilities, support
        )
    else:
        # Compute the target Q-value
        next_q_values = jnp.squeeze(target_network_dist.q_values)
        replay_next_qt_max = next_q_values[next_qt_argmax]
        target = rewards + gamma_with_terminal * replay_next_qt_max

    return jax.lax.stop_gradient(target)


class SimHash:
    """
    SimHash class is used for generating a locality-sensitive hash for input states.
    It uses random hyperplanes to project states and compute binary hashes.
    The class also tracks the occurrence of each hash to provide an exploration bonus.
    """

    def __init__(
            self, input_dimension, num_hyperplanes, key=jax.random.PRNGKey(0)
    ):
        # Compute flattened dimension for a single frame, regardless of the frame_stack_size
        single_frame_dim = jnp.prod(jnp.array(input_dimension))

        key, subkey = jax.random.split(key)

        self.num_hyperplanes = num_hyperplanes
        self.hyperplanes = jax.random.normal(
            subkey, (num_hyperplanes, single_frame_dim)
        )
        self.counts = {}  # Dictionary to store counts of each hash
        self.key = key

    @staticmethod
    @jax.jit
    def _compute_hash_and_bonus(state, hyperplanes):
        flattened_state = state.flatten()
        binary_hash = (jnp.dot(hyperplanes, flattened_state) > 0).astype(int)

        # Convert binary hash to an integer
        hash_val = jnp.dot(2 ** jnp.arange(binary_hash.shape[0]), binary_hash)

        return hash_val

    def get_hash_and_bonus(self, state, beta=1.0):
        hash_val = int(
            self._compute_hash_and_bonus(state, self.hyperplanes).item()
        )

        # Retrieve and update counts
        count = self.counts.get(hash_val, 0) + 1
        self.counts[hash_val] = count
        state_bonus = beta / jnp.sqrt(count)

        return state_bonus, hash_val


class UnifiedHomeostasis:
    """
    A class to manage the homeostasis of different triggers.
    It maintains moving averages for these triggers and computes exploration probabilities
    based on the deviations from the averages.
    """

    # Define a namedtuple to hold values associated with each trigger type
    MovingAverageData = namedtuple(
        "MovingAverageData", ["mean", "var", "mean_pos"]
    )

    def __init__(self, target_rate, key=jax.random.PRNGKey(1)):
        self.target_rate = target_rate
        self.t = 0
        self.epsilon = 1e-8  # for numerical stability
        self.trigger_types = [
            "vpd",
            "state_bonus",
        ]  # configurable trigger types
        self._initialize_moving_averages()
        self.key = key

    def _initialize_moving_averages(self):
        # Initialize a single dictionary to hold all moving average related data
        self.data = {
            etype: self.MovingAverageData(0, 0, 0)
            for etype in self.trigger_types
        }

    def _get_tau(self):
        """Computes the time constant for the moving averages."""
        return min(self.t, 5 / self.target_rate) + self.epsilon

    def update_target_rate(self, new_target_rate):
        self.target_rate = new_target_rate

    def update_moving_averages(self, x, trigger_type):
        data = self.data[trigger_type]
        (
            new_mean,
            new_var,
            new_mean_pos,
            x_pos,
        ) = UnifiedHomeostasis._update_moving_averages_jit(
            x, data, self.t, self.target_rate, self.epsilon
        )
        self.data[trigger_type] = self.MovingAverageData(
            new_mean, new_var, new_mean_pos
        )
        return x_pos

    def compute_exploration_probabilities(self, *trigger_values):
        self.t += 1
        probabilities = []

        for idx, value in enumerate(trigger_values):
            if value is None:
                continue
            x_pos = self.update_moving_averages(value, self.trigger_types[idx])
            prob = UnifiedHomeostasis._compute_exploration_prob_jit(
                x_pos,
                self.data[self.trigger_types[idx]].mean_pos,
                self.target_rate,
                self.epsilon,
            )
            probabilities.append(prob)

        return tuple(probabilities)

    def combine_decisions(self, *probs):
        """Returns the final exploration probability and a Bernoulli sample."""
        combined_prob = jnp.sum(jnp.array(probs)) / len(probs)
        self.key, subkey = jax.random.split(self.key)

        return combined_prob, jax.random.bernoulli(subkey, float(combined_prob))

    def reset(self):
        """Reset moving averages and time."""
        self._initialize_moving_averages()
        self.t = 0

    @staticmethod
    @jax.jit
    def _update_moving_averages_jit(x, data, t, target_rate, epsilon):
        tau = jnp.minimum(t, 5 / target_rate) + epsilon
        alpha = 1 / tau
        new_mean = (1 - alpha) * data.mean + alpha * x
        new_var = (1 - alpha) * data.var + alpha * (x - new_mean) ** 2
        x_standardized = (x - new_mean) / (jnp.sqrt(new_var) + epsilon)
        x_pos = jnp.exp(x_standardized)
        new_mean_pos = (1 - alpha) * data.mean_pos + alpha * x_pos
        return new_mean, new_var, new_mean_pos, x_pos

    @staticmethod
    @jax.jit
    def _compute_exploration_prob_jit(x_pos, mean_pos, target_rate, epsilon):
        return jnp.minimum(1, target_rate * (x_pos / (mean_pos + epsilon)))


def _compute_discounted_rewards(rewards, gamma):
    """Helper function to compute the sum of discounted rewards."""
    return sum(gamma ** i * reward for i, reward in enumerate(rewards))


class ValueStorage:
    """
    ValueStorage maintains a buffer of the latest k values and rewards.
    It allows querying for various calculations like Value Prediction Difference (VPD) and
    retrieving the latest value estimate.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Nested named tuple definition
    Entry = namedtuple("Entry", ["value", "reward"])

    def add(self, value, reward):
        """Add a new value and reward to the buffer."""
        self.buffer.append(self.Entry(value, reward))

    def latest_value_estimate(self):
        """Retrieve the latest value estimate."""
        return self.buffer[-1].value if self.buffer else None

    def get_promised_value_and_discounted_rewards(self, k, gamma):
        """Calculate the promised value and sum of discounted rewards for the last k entries."""
        if len(self.buffer) <= k:
            return None, None

        # Extract the values and rewards for the last k entries
        last_k_entries = list(self.buffer)[-k:]
        promised_value = last_k_entries[0].value
        rewards = [entry.reward for entry in last_k_entries]

        discounted_rewards = _compute_discounted_rewards(rewards, gamma)

        return promised_value, discounted_rewards

    def calculate_vpd(self, k, gamma, current_value):
        """Compute the Value Prediction Difference (VPD) for the current value."""
        (
            promised_value,
            discounted_rewards,
        ) = self.get_promised_value_and_discounted_rewards(k, gamma)

        if promised_value is None:
            return jnp.array(0.0)

        promised_value = jnp.squeeze(promised_value)
        current_value = jnp.squeeze(current_value)

        target_value = discounted_rewards + (gamma ** k) * current_value
        vpd = jnp.abs(target_value - promised_value)

        return vpd

    def reset(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self):
        """Return the number of entries in the buffer."""
        return len(self.buffer)


@gin.configurable
class JaxFullRainbowAgent(dqn_agent.JaxDQNAgent):
    """A compact implementation of the full Rainbow agent."""

    def __init__(
            self,
            num_actions,
            observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
            observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
            stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
            pickle_output=False,
            use_vpd=False,
            use_simhash=False,
            vpd_k=3,
            boltzmann=False,
            ez_greedy=False,
            noisy=True,
            dueling=True,
            double_dqn=True,
            distributional=True,
            num_updates_per_train_step=1,
            network=networks.FullRainbowNetwork,
            num_atoms=51,
            vmax=10.0,
            vmin=None,
            epsilon_fn=dqn_agent.linearly_decaying_epsilon,
            replay_scheme="prioritized",
            summary_writer=None,
            seed=None,
            preprocess_fn=None,
    ):
        """Initializes the agent and constructs the necessary components.

        Args:
                num_actions: int, number of actions the agent can take at any state.
                noisy: bool, Whether to use noisy networks or not.
                dueling: bool, Whether to use dueling network architecture or not.
                double_dqn: bool, Whether to use Double DQN or not.
                distributional: bool, whether to use distributional RL or not.
                num_updates_per_train_step: int, Number of gradient updates every training
                        step. Defaults to 1.
                network: flax.linen Module, neural network used by the agent initialized
                        by shape in _create_network below. See
                        dopamine.jax.networks.RainbowNetwork as an example.
                num_atoms: int, the number of buckets of the value function distribution.
                vmax: float, the value distribution support is [vmin, vmax].
                vmin: float, the value distribution support is [vmin, vmax]. If vmin is
                        None, it is set to -vmax.
                epsilon_fn: function expecting 4 parameters: (decay_period, step,
                        warmup_steps, epsilon). This function should return the epsilon value
                        used for exploration during training.
                replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
                        replay memory.
                summary_writer: SummaryWriter object, for outputting training statistics.
                seed: int, a seed for Jax RNG and initialization.
                preprocess_fn: function expecting the input state as parameter which
                        it preprocesses (such as normalizing the pixel values between 0 and 1)
                        before passing it to the Q-network. Defaults to None.
        """
        logging.info(
            "Creating %s agent with the following parameters:",
            self.__class__.__name__,
        )
        logging.info("\t use_simhash: %s", use_simhash)
        logging.info("\t use_vpd: %s", use_vpd)
        if use_vpd:
            logging.info("\t vpd_k: %s", vpd_k)
        logging.info("\t boltzmann: %s", boltzmann)
        logging.info("\t ez_greedy: %s", ez_greedy)
        logging.info("\t noisy_networks: %s", noisy)
        logging.info("\t double_dqn: %s", double_dqn)
        logging.info("\t dueling_dqn: %s", dueling)
        logging.info("\t distributional: %s", distributional)
        logging.info("\t num_atoms: %d", num_atoms)
        logging.info("\t replay_scheme: %s", replay_scheme)
        logging.info(
            "\t num_updates_per_train_step: %d", num_updates_per_train_step
        )
        # We need this because some tools convert round floats into ints.
        vmax = float(vmax)
        self._num_atoms = num_atoms
        vmin = vmin if vmin else -vmax
        self._support = jnp.linspace(vmin, vmax, num_atoms)
        self._replay_scheme = replay_scheme
        self._double_dqn = double_dqn
        self._noisy = noisy
        self._dueling = dueling
        self._distributional = distributional
        self._num_updates_per_train_step = num_updates_per_train_step

        super().__init__(
            num_actions=num_actions,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            stack_size=stack_size,
            network=functools.partial(
                network,
                num_atoms=num_atoms,
                noisy=self._noisy,
                dueling=self._dueling,
                distributional=self._distributional,
            ),
            epsilon_fn=zero_epsilon if self._noisy else epsilon_fn,
            summary_writer=summary_writer,
            seed=seed,
            preprocess_fn=preprocess_fn,
        )

        self.pickle_output = pickle_output

        if boltzmann:
            self.exploration_strategy = Strategy.BOLTZMANN
        elif ez_greedy:
            self.exploration_strategy = Strategy.EZ_GREEDY
        else:
            self.exploration_strategy = Strategy.E_GREEDY

        target_rate = self.epsilon_train
        self.homeostasis = UnifiedHomeostasis(target_rate)

        self.exploration_steps_left = 0
        self.last_action = -1

        self.use_simhash = use_simhash
        self.use_vpd = use_vpd
        self.vpd_k = vpd_k

        self.use_trigger = self.use_simhash or self.use_vpd

        self.value_storage = ValueStorage(10)

        self.logging_data = {
            "vpds": [],
            "state_bonuses": [],
            "exploration_decisions": [],
        }

        if stack_size > 1:
            # If we are using a stack of frames, we usualy will need
            # to use more hyperplanes, such as for Atari examples
            num_hyperplanes = 256
        else:
            # Rest of the environments can use a smaller number of hyperplanes
            # Classic control for instance
            num_hyperplanes = 16

        self.state_counts = SimHash(observation_shape, num_hyperplanes)

    def _build_networks_and_optimizer(self):
        self._rng, rng = jrandom.split(self._rng)
        self.online_params = self.network_def.init(
            rng, x=self.state, support=self._support
        )
        self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = self.online_params

    def _build_replay_buffer(self):
        """Creates the replay buffer used by the agent."""
        if self._replay_scheme not in ["uniform", "prioritized"]:
            raise ValueError(
                "Invalid replay scheme: {}".format(self._replay_scheme)
            )
        return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype,
        )

    def _training_step_update(self):
        """Gradient update during every training step."""

        self._sample_from_replay_buffer()
        states = self.preprocess_fn(self.replay_elements["state"])
        next_states = self.preprocess_fn(self.replay_elements["next_state"])

        if self._replay_scheme == "prioritized":
            # The original prioritized experience replay uses a linear exponent
            # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
            # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
            # suggested a fixed exponent actually performs better, except on Pong.
            probs = self.replay_elements["sampling_probabilities"]
            # Weight the loss by the inverse priorities.
            loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
            loss_weights /= jnp.max(loss_weights)
        else:
            # Uniform weights if not using prioritized replay.
            loss_weights = jnp.ones(states.shape[0])

        (
            self.optimizer_state,
            self.online_params,
            loss,
            mean_loss,
            self._rng,
        ) = train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            states,
            self.replay_elements["action"],
            next_states,
            self.replay_elements["reward"],
            self.replay_elements["terminal"],
            loss_weights,
            self._support,
            self.cumulative_gamma,
            self._double_dqn,
            self._distributional,
            self._rng,
        )

        if self._replay_scheme == "prioritized":
            # Rainbow and prioritized replay are parametrized by an exponent
            # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
            # leave it as is here, using the more direct sqrt(). Taking the square
            # root "makes sense", as we are dealing with a squared loss.  Add a
            # small nonzero value to the loss to avoid 0 priority items. While
            # technically this may be okay, setting all items to 0 priority will
            # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
            self._replay.set_priority(
                self.replay_elements["indices"], jnp.sqrt(loss + 1e-10)
            )

        if (
                self.summary_writer is not None
                and self.training_steps > 0
                and self.training_steps % self.summary_writing_frequency == 0
        ):
            with self.summary_writer.as_default():
                tf.summary.scalar(
                    "CrossEntropyLoss", mean_loss, step=self.training_steps
                )
            self.summary_writer.flush()
            if hasattr(self, "collector_dispatcher"):
                self.collector_dispatcher.write(
                    [
                        statistics_instance.StatisticsInstance(
                            "Loss",
                            onp.asarray(mean_loss),
                            step=self.training_steps,
                        ),
                    ],
                    collector_allowlist=self._collector_allowlist,
                )

    def _store_transition(
            self,
            last_observation,
            action,
            reward,
            is_terminal,
            *args,
            priority=None,
            episode_end=False,
    ):
        """Stores a transition when in training mode."""
        is_prioritized = isinstance(
            self._replay,
            prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer,
        )
        if is_prioritized and priority is None:
            if self._replay_scheme == "uniform":
                priority = 1.0
            else:
                priority = self._replay.sum_tree.max_recorded_priority

        if not self.eval_mode:
            self._replay.add(
                last_observation,
                action,
                reward,
                is_terminal,
                *args,
                priority=priority,
                episode_end=episode_end,
            )

    def _train_step(self):
        """Runs a single training step.

        Runs training if both:
                (1) A minimum number of frames have been added to the replay buffer.
                (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online_network_params to target_network_params if
        training steps is a multiple of target update period.
        """
        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                for _ in range(self._num_updates_per_train_step):
                    self._training_step_update()

            if self.training_steps % self.target_update_period == 0:
                self._sync_weights()

        self.training_steps += 1

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode."""
        if self.pickle_output:
            update_pickle_file(
                "../../tmp/exploration_data.pkl", self.logging_data
            )
            self.logging_data = {
                "vpds": [],
                "state_bonuses": [],
                "exploration_decisions": [],
            }

        self._reset_state()
        self._record_observation(observation)

        self.value_storage.reset()

        if not self.eval_mode:
            self._train_step()

        state = self.preprocess_fn(self.state)

        self._rng, subkey = jrandom.split(self._rng)

        # Default values for beginning of an episode
        self.exploration_steps_left, self.last_action = 0, -1
        exploration_signal = False
        explore_duration = 0

        (
            self._rng,
            self.exploration_steps_left,
            self.action,
            state_value,
            was_exploratory_decision,
            epsilon
        ) = select_action(
            self.network_def,
            self.online_params,
            state,
            subkey,
            self.num_actions,
            self.eval_mode,
            self.epsilon_eval,
            self.epsilon_train,
            self.epsilon_decay_period,
            self.training_steps,
            self.min_replay_history,
            self.epsilon_fn,
            self._support,
            self.exploration_strategy,
            self.exploration_steps_left,
            self.last_action,
            self.use_trigger,
            exploration_signal,
            explore_duration,
        )

        self.homeostasis.update_target_rate(epsilon)

        if self.pickle_output:
            self.logging_data["exploration_decisions"].append(
                int(was_exploratory_decision)
            )

        # State counts only based on last frame for atari environments
        if len(state.shape) > 2 and state.shape[-1] > 1:
            last_frame = state[:, :, -1]
        else:
            last_frame = state

        if self.use_simhash:
            self.state_counts.get_hash_and_bonus(last_frame)
        if self.use_vpd:
            self.value_storage.add(value=state_value, reward=0)

        self.last_action = self.action

        return int(self.action)

    def step(self, reward, observation):
        """Records reward and observation, returns the agent's next action."""
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(
                self._last_observation, self.action, reward, False
            )
            self._train_step()

        state = self.preprocess_fn(self.state)

        if self.use_trigger:
            state_bonus, vpd = None, None

            # State counts only based on last frame for atari environments
            if len(state.shape) > 2 and state.shape[-1] > 1:
                last_frame = state[:, :, -1]
            else:
                last_frame = state

            state_bonus, state_hash = self.state_counts.get_hash_and_bonus(
                last_frame
            )
            if self.use_vpd:
                latest_value = self.value_storage.latest_value_estimate()
                vpd = self.value_storage.calculate_vpd(
                    self.vpd_k, self.gamma, latest_value
                )

            # Calculate exploration probability based on state bonus and/or vpd
            trigger_probabilities = (
                self.homeostasis.compute_exploration_probabilities(
                    vpd, state_bonus
                )
            )
            # Get exploration signal
            (
                combined_prob,
                exploration_signal,
            ) = self.homeostasis.combine_decisions(*trigger_probabilities)
            # Log exploration data
            if self.pickle_output:
                self.logging_data["state_bonuses"].append(
                    float(state_bonus) if state_bonus else 0.0
                )
                self.logging_data["vpds"].append(float(vpd) if vpd else 0.0)
        else:
            exploration_signal = None

        # Extended exploration only for ez-greedy
        explore_duration = (
            jnp.minimum(zipf(2.0), 100)
            if self.exploration_strategy == Strategy.EZ_GREEDY
            else 0
        )

        # Splitting RNG key for select_action
        self._rng, subkey_for_select = jax.random.split(self._rng)
        (
            self._rng,
            self.exploration_steps_left,
            self.action,
            state_value,
            was_exploratory_decision,
            epsilon,
        ) = select_action(
            self.network_def,
            self.online_params,
            state,
            subkey_for_select,
            self.num_actions,
            self.eval_mode,
            self.epsilon_eval,
            self.epsilon_train,
            self.epsilon_decay_period,
            self.training_steps,
            self.min_replay_history,
            self.epsilon_fn,
            self._support,
            self.exploration_strategy,
            self.exploration_steps_left,
            self.last_action,
            self.use_trigger,
            exploration_signal,
            explore_duration,
        )

        self.homeostasis.update_target_rate(epsilon)

        if self.pickle_output:
            self.logging_data["exploration_decisions"].append(
                int(was_exploratory_decision)
            )

        self.last_action = self.action

        if self.use_vpd:
            self.value_storage.add(value=state_value, reward=reward)

        return int(self.action)


def update_pickle_file(filename, data):
    # Load existing data if the file exists
    if os.path.exists(filename) and data:
        with open(filename, "rb") as f:
            all_data = pickle.load(f)
    else:
        all_data = []

    # Append new data to existing data
    if data:  # Check if data list is not empty
        all_data.append(data)

    # Save updated data back to file
    with open(filename, "wb") as f:
        pickle.dump(all_data, f)
