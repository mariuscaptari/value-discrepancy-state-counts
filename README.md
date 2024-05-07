# VDSC: Enhancing Exploration Timing with Value Discrepancy and State Counts

Value Discrepancy and State Counts, or VDSC for short, is an exploration strategy that leverage the DRL agent's internal state to decide _when_ to explore, addressing the shortcomings of blind switching mechanisms.

The code provided here is an extension of the Dopamine framework, stripped of unnecessary components. [Dopamine](https://arxiv.org/abs/1812.06110) is a research framework for fast prototyping of reinforcement learning algorithms.

The Rainbow ([Hessel et al., 2018][rainbow]) agent (JAX implementation) was used across all experiments.

## Prerequisites

Install the necessary Atari environments before installing the rest of the packages:

**Atari**

1. Install the atari roms following the instructions from
[atari-py](https://github.com/openai/atari-py#roms).
2. `pip install ale-py`
3. `unzip $ROM_DIR/ROMS.zip -d $ROM_DIR && ale-import-roms $ROM_DIR/ROMS`
(replace $ROM_DIR with the directory you extracted the ROMs to).

**Install necessary packages**

`pip install -r requirements.txt`

## Running tests

You can test whether the installation was successful by running the following
from the root directory.

```
export PYTHONPATH=$PYTHONPATH:$PWD
python -m tests.dopamine.atari_init_test
```

## Running the experiements

All experiments were ran on a SLURM cluster using the following command from the root directory to batch the jobs:

`sbatch job_atari.sh <game> <method> <iteration_number>`

where:
- `game` is the name of the Atari game (e.g., Freeway, Frostbite, ...)
- `method` is one of the following implemented exploration methods: `e-greedy`, `ez-greedy`, `boltzmann`, `noisy`, `vpd`, `simhash`, `vpd_sim_hash`
- `iteration` refers to the numbered seed run (e.g., 0, 1, 2, ...)

Additionaly, the `job_atari_exploration.sh` script provides a quick way to get a summary of the current training progress for each of the Atari games.

## Plot the results

Two sepparate notebooks can be found in the `dopamine/colab` directory containing code to generate plots for vizualizing learning curves, as well as exploratory behaviour.

## References

[Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning*.
Proceedings of the AAAI Conference on Artificial Intelligence, 2018.][rainbow]

[rainbow]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17204/16680

