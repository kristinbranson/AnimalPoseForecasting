# Animal Pose Forecasting Library

| **Real trajectories** | **Agent-centric simulation** |
|:---:|:---:|
| ![Trajectories of real flies](images/gt_sample1_640p.gif) | ![Simulated trajectories from agent-centric model](images/ref_sample1_640p.gif) |

> 🚧👷‍♀️ **Under construction** — this library and its documentation are a work in progress. 🏗️🚧

The Animal Pose Forecasting library (APF) contains code for training, simulating from, and evaluating **agent-centric** autoregressive models of **animal behavior** from tracked **pose**. Unlike standard world-frame approaches, agent-centric models input egocentric sensory observations and output egocentric movements, mirroring the biological constraint that animals observe and act on the world from their own reference frame. Social behavior emerges from agents independently sensing and responding to one another. 

This Python and PyTorch-based library is built around the observation that, to train and generate from agent-centric models, the **same data** needs to be available in **many representations**: world-frame, egocentric, sensory, ML-friendly. Translating between these representations requires **composing sequences of operations** or their inverses. Our library makes these operations explicit, manipulable objects that are composable, invertible, and serializable alongside the model. The sequence of operations applied to a chunk of data is itself stored, so the library knows how a given representation was constructed and can invert or serialize it on demand.

![Overview of the APF library. (a) World-frame vs agent-centric frames. (b) Examples of hand-designed, egocentric sensory inpputs. (c) Inputting egocentric sensory information and outputting eogcentric movements parallels the animal’s own sensory-to-motor mapping. (d) APF composes biologically motivated (solid purple) and ML-motivated (outlined) transformations from world-frame keypoints to model inputs and labels, and inverts the chain at rollout to produce the next steps sensory inputs](images/overview.png)

## Model variants

For each variant, the flow diagram (left) shows how data is transformed, and the GIF (right) shows a simulation from that model. Colored circles highlight simulated male flies; non-highlighted flies are real female flies. 

| Operations | Simulation |
|:---:|:---:|
| <img src="images/flow_ref.png" height="640" alt="Flow diagram for ref variant"> | <img src="images/ref_sample1_640p.gif" height="640" alt="Simulation from ref variant"> |
| <img src="images/flow_rawkp.png" height="640" alt="Flow diagram for rawkp variant"> | <img src="images/rawkp_sample1_640p.gif" height="640" alt="Simulation from rawkp variant"> |
| <img src="images/flow_bodycentric.png" height="640" alt="Flow diagram for bodycentric variant"> | <img src="images/bodycentric_sample1_640p.gif" height="640" alt="Simulation from bodycentric variant"> |
| <img src="images/flow_binall.png" height="640" alt="Flow diagram for binall variant"> | <img src="images/binall_sample1_640p.gif" height="640" alt="Simulation from binall variant"> |
| <img src="images/flow_nobin.png" height="640" alt="Flow diagram for nobin variant"> | <img src="images/nobin_sample1_640p.gif" height="640" alt="Simulation from nobin variant"> |
| <img src="images/flow_predpose.png" height="640" alt="Flow diagram for predpose variant"> | <img src="images/predpose_sample1_640p.gif" height="640" alt="Simulation from predpose variant"> |
| <img src="images/flow_short.png" height="640" alt="Flow diagram for short variant"> | <img src="images/short_sample1_640p.gif" height="640" alt="Simulation from short variant"> |

## Credits

Library developed by [Eyrun Eyjolfsdottir](https://github.com/eyrun) and [Kristin Branson](https://github.com/kristinbranson)