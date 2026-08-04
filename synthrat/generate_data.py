# load data
import apf_ratinabox
import pickle
import numpy as np
import matplotlib.pyplot as plt

# load state
with open('data/ratinabox_rl_state_20260423.pkl', 'rb') as f:
    data = pickle.load(f)
    
# rehydrate everything
res = apf_ratinabox.rehydrate_data(data)
Env = res['Env']
Ag = res['Ag']
Inputs = res['Inputs']
Reward = res['Reward']
ValNeur = res['ValNeur']
Sensory = res['Sensory']

# generate episodes
episodes = apf_ratinabox.generate_episodes(
    Ag, Env, Inputs, Reward, ValNeur,
    nepisodes=10000, nworkers=32, seed=0,
    exploit_explore_ratio=1.0, episode_end_time=3.0,
    max_t=60, framerate=None,
)
data['track'] = episodes['track']
data['hidden'] = episodes['hidden']

# save data
with open('data/ratinabox_rl_data_20260423.pkl', 'wb') as f:
    pickle.dump(data, f)
    
# visualize sensory input for one episode
ep = 50
t = 50
sensory_curr = apf_ratinabox.compute_sensory(data['track'][ep], Sensory)
apf_ratinabox.visualize_sensory(data['track'][ep], sensory_curr, t=t, Env=Env, Sensory=Sensory)

# visualize trajectories for multiple episodes
nplot = 9
nc = int(np.ceil(np.sqrt(nplot)))
nr = int(np.ceil(nplot / nc))
fig,ax = plt.subplots(nr, nc, sharex=True, sharey=True, figsize=(5*nc, 5*nr))
ax = ax.flatten()
for ep in range(nplot):
    apf_ratinabox.plot_episode(episodes['track'][ep], Env, axcurr=ax[ep])