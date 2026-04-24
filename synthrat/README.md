Code for generating data from the RatInABox synthetic rat model
- `ratinabox/`: git repo from RatInABox work
- `freeze_reinforcement_learning_example.*`: runs the RL example notebook from `ratinabox/demos` then saves the state to `data/ratinabox_rl_state_<timestamp>.pkl`
- `generate_data.py`: generate many episodes from the saved state and save it to 
- `apf_ratinabox.py`: Code that is useful both for
  - Generating and saving episodes
  - Using generated data with APF
