# load data
import apf_ratinabox
import pickle
import numpy as np

import matplotlib
matplotlib.use('TkAgg') 

import matplotlib.pyplot as plt
import os

plt.ion()

CHANGE_SENSORY = True
DOGENERATE = True
VISUALIZE = True
HISTOGRAM = True

statefile = 'ratinabox_rl_state_20260423.pkl'
savestatefile = 'ratinabox_rl_state_20260423.pkl'
savetimestamp = '20260423'

nworkers = 32
ntrain_episodes = 10000
nval_episodes = 1000
seed_train = 0
seed_val = 1
exploit_explore_ratio = 1.0
episode_end_time = 3.0
max_t = 60
framerate = None

thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(thisdir, 'data')
figdir = os.path.join(thisdir, 'figures')
os.makedirs(datadir, exist_ok=True)
os.makedirs(figdir, exist_ok=True)

# load state
with open(os.path.join(datadir, statefile), 'rb') as f:
    data = pickle.load(f)
    
# rehydrate everything
res = apf_ratinabox.rehydrate_data(data)
Env = res['Env']
Ag = res['Ag']
Inputs = res['Inputs']
Reward = res['Reward']
ValNeur = res['ValNeur']
Sensory = res['Sensory']

# change sensory
if CHANGE_SENSORY:
    n_boundary_vector_cells = 16
    boundary_distance_range = (0.02, 0.4)
    boundary_angle_range = 150
    boundary_spatial_resolution = 0.02   # radial cell size at innermost ring (m)
    boundary_beta = 5                    # Hartley growth parameter; smaller -> fewer rings
    n_head_direction_cells = 16
    angular_spread_range = (10, 30)

    cell_config = {
        # "boundary_vector": {
        #     "type": "BoundaryVectorCells",
        #     "n": n_boundary_vector_cells,
        #     "reference_frame": "egocentric",                   # wall positions relative to heading
        #     "tuning_distance": boundary_distance_range,        # covers near walls + far wall of unit env
        #     "angular_spread": angular_spread_range,            # degrees; default
        #     "tuning_angle": np.linspace(-boundary_angle_range, boundary_angle_range, n_boundary_vector_cells, endpoint=False),
        # },
        "field_of_view_boundary":   {
            "type": "FieldOfViewBVCs",
            "cell_arrangement": 'rect_polar_grid',
            "distance_range": boundary_distance_range,
            "angle_range": [0,boundary_angle_range],
            "n_angles": n_boundary_vector_cells,
            "spatial_resolution": boundary_spatial_resolution,
            "beta": boundary_beta,
        },
        "head_direction": {
            "type": "HeadDirectionCells",
            "n": n_head_direction_cells,                       # 36° bins around the circle
            "angular_spread_degrees": angular_spread_range[1], # von-Mises width
        },
        "speed": {
            "type": "SpeedCell",                               # n is forced to 1
        },
    }
    data['sensory_config'] = cell_config

    Sensory = apf_ratinabox.init_sensory(Ag, Env, cell_config)
    data['sensory_info'] = apf_ratinabox.get_sensory_info(Sensory)

    print('Changed sensory config to: ' + str(cell_config))
    print('Writing new state with updated sensory config to file: ' + os.path.join(datadir, savestatefile))
    with open(os.path.join(datadir, savestatefile), 'wb') as f:
        pickle.dump(data, f)

if DOGENERATE:

    # generate train episodes
    print('Generating train episodes...')
    episodes = apf_ratinabox.generate_episodes(
        Ag, Env, Inputs, Reward, ValNeur,
        nepisodes=ntrain_episodes, nworkers=nworkers, seed=seed_train,
        exploit_explore_ratio=exploit_explore_ratio, episode_end_time=episode_end_time,
        max_t=max_t, framerate=framerate,
    )
    data['track'] = episodes['track']
    data['hidden'] = episodes['hidden']

    # save data
    print('Saving train episodes to file: ' + os.path.join(datadir, f'ratinabox_rl_traindata_{savetimestamp}.pkl'))
    with open(os.path.join(datadir, f'ratinabox_rl_traindata_{savetimestamp}.pkl'), 'wb') as f:
        pickle.dump(data, f)
        
    # generate val episodes
    print('Generating val episodes...')
    episodes = apf_ratinabox.generate_episodes(
        Ag, Env, Inputs, Reward, ValNeur,
        nepisodes=nval_episodes, nworkers=nworkers, seed=seed_val,
        exploit_explore_ratio=exploit_explore_ratio, episode_end_time=episode_end_time,
        max_t=max_t, framerate=framerate,
    )
    data['track'] = episodes['track']
    data['hidden'] = episodes['hidden']
        
    # save data
    print('Saving val episodes to file: ' + os.path.join(datadir, f'ratinabox_rl_valdata_{savetimestamp}.pkl'))
    with open(os.path.join(datadir, f'ratinabox_rl_valdata_{savetimestamp}.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
else:
    # load val episodes
    print('Loading val episodes from file: ' + os.path.join(datadir, f'ratinabox_rl_valdata_{savetimestamp}.pkl'))
    with open(os.path.join(datadir, f'ratinabox_rl_valdata_{savetimestamp}.pkl'), 'rb') as f:
        data = pickle.load(f)
    episodes = {'track': data['track'], 'hidden': data['hidden']}


if 'field_of_view_boundary' in Sensory and data['sensory_info']['field_of_view_boundary']['cell_arrangement'] == 'rect_polar_grid':
    n_rings,n_neurons_per_ring = apf_ratinabox.get_rect_polar_grid_shape(Sensory['field_of_view_boundary'])
    print(f"Field of view boundary cells are arranged in a rect-polar grid with {n_rings} rings and {n_neurons_per_ring} neurons per ring.")
    
if VISUALIZE:
    # visualize sensory input for one episode
    print('Visualizing sensory input for one episode...')
    nplot = 3
    nc = 1+len(Sensory)
    fig,ax = plt.subplots(nplot,nc,figsize=(5*nc,5*nplot),sharex='col',sharey='col')
    eps = np.arange(nplot)
    fract = np.linspace(0,1,nplot+2)[1:-1]
    for i,ep in enumerate(eps):
        t = int(fract[i]*episodes['track'][ep]['pos'].shape[0])
        axcurr = {'traj': ax[i,0]}
        sensory_curr = apf_ratinabox.compute_sensory(episodes['track'][ep], Sensory)
        for sensorkey in Sensory.keys():
            axcurr[sensorkey] = ax[i,1+list(Sensory.keys()).index(sensorkey)]
        apf_ratinabox.visualize_sensory(episodes['track'][ep], sensory_curr, t, Env, Sensory, ax=axcurr, fig=fig)
        axcurr['traj'].set_title(f'episode {ep}, t={t}')
    # save figure
    plt.savefig(os.path.join(figdir, f'sensory_visualization_ep_{savetimestamp}.png'))
    plt.savefig(os.path.join(figdir, f'sensory_visualization_ep_{savetimestamp}.pdf'))

    # visualize trajectories for multiple episodes
    print('Visualizing trajectories for multiple episodes...')
    nplot = 9
    nc = int(np.ceil(np.sqrt(nplot)))
    nr = int(np.ceil(nplot / nc))
    fig,ax = plt.subplots(nr, nc, sharex=True, sharey=True, figsize=(5*nc, 5*nr))
    ax = ax.flatten()
    for ep in range(nplot):
        apf_ratinabox.plot_episode(episodes['track'][ep], Env, axcurr=ax[ep])
    plt.savefig(os.path.join(figdir, f'example_trajectories_{savetimestamp}.png'))
    plt.savefig(os.path.join(figdir, f'example_trajectories_{savetimestamp}.pdf'))

# histogram velocities and accelerations computed from track
if HISTOGRAM:

    episodes_discrete_vel = {
        'foward_vel': [],
        'sideways_vel': [],
        'orientation_vel': [],
    }
    for episode in episodes['track']:
        forward_vel, sideways_vel, orientation_vel = apf_ratinabox.compute_velocity(episode)
        episodes_discrete_vel['foward_vel'].append(forward_vel)
        episodes_discrete_vel['sideways_vel'].append(sideways_vel)
        episodes_discrete_vel['orientation_vel'].append(orientation_vel)

    fig,ax = plt.subplots(3,3,figsize=(15,15))
    ps = np.linspace(0,100,26)
    for i,(k,v) in enumerate(episodes_discrete_vel.items()):
        ax[0,i].hist(np.concatenate(v), bins=50)
        ax[0,i].set_title('Histogram of ' + k)
        prctiles = np.percentile(np.concatenate(v), ps)
        ax[1,i].plot(ps, prctiles,'.-')
        ax[1,i].set_title(k)
        ax[2,i].plot((prctiles[:-1]+prctiles[1:])/2, np.diff(prctiles),'.-')
        ax[2,i].set_title(k + ' prctile diff')
        ax[2,i].set_yscale('log')

    fig.tight_layout()
    plt.savefig(os.path.join(figdir, f'velocity_histograms_{savetimestamp}.png'))
    plt.savefig(os.path.join(figdir, f'velocity_histograms_{savetimestamp}.pdf'))

plt.ioff()
plt.show()