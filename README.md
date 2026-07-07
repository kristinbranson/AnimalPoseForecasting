# Animal Pose Forecasting Library

> 🚧👷‍♀️ **Under construction** — this library and its documentation are a work in progress. 🏗️🚧

| **Real trajectories** | **Agent-centric simulation** | **World-frame simulation** |
|:---:|:---:|:---:|
| ![Trajectories of real flies](images/gt_sample1_640p.gif) | ![Simulated trajectories from agent-centric model](images/ref_sample1_640p.gif) | ![Simulated trajectories from world-reference model](images/rawkp_sample1_640p.gif) |

![Overview of the APF library. (a) World-frame vs agent-centric frames. (b) Examples of hand-designed, egocentric sensory inpputs. (c) Inputting egocentric sensory information and outputting eogcentric movements parallels the animal’s own sensory-to-motor mapping. (d) APF composes biologically motivated (solid purple) and ML-motivated (outlined) transformations from world-frame keypoints to model inputs and labels, and inverts the chain at rollout to produce the next steps sensory inputs](images/overview.png)

The Animal Pose Forecasting library (APF) contains code for training, simulating from, and evaluating **agent-centric** autoregressive models of **animal behavior** from tracked **pose**. Unlike standard world-frame approaches, agent-centric models input egocentric sensory observations and output egocentric movements, mirroring the biological constraint that animals observe and act on the world from their own reference frame. Social behavior emerges from agents independently sensing and responding to one another. 

This Python and PyTorch-based library is built around the observation that, to train and generate from agent-centric models, the **same data** needs to be available in **many representations**: world-frame, egocentric, sensory, ML-friendly. Translating between these representations requires **composing sequences of operations** or their inverses. Our library makes these operations explicit, manipulable objects that are composable, invertible, and serializable alongside the model. The sequence of operations applied to a chunk of data is itself stored, so the library knows how a given representation was constructed and can invert or serialize it on demand.

