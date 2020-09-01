
----

## Overview

![Architecture](./img/method.png "")

To solve tasks in obstructed environments, we propose Motion Planner Augmented Reinforcement Learning (<b>MoPA-RL</b>). Our framework consists of a <b>RL policy</b> and a <b>motion planner</b>. The motion planner is integrated into the RL policy by enlarging the action space. If a sampled action from the RL policy is in the original action space, an agent directly executes the action to the environment, otherwise the motion planner computes a path to move the agent to faraway points. MoPA-RL has three benefits:
<ul>
<li>Able to add motion planning capabilities to any RL agent with joint space control</li>
<li>Allow an agent to freely switch between MP and direct action execution  by controlling the scale of action</li>
<li>Naturally learns trajectories that avoid collisions by leveraging motion planning </li>
</ul>

----

## Videos

<span class="env-name"><b>Sawyer Push</b></span>
- Sawyer arm is required to find a path to reach an object inside of a box, and push it to a goal position.
<div class="w3-row-padding">
	<div class="w3-col s1 w3-center">
	</div>
	<div class="w3-col s5 w3-center">
		<video height="auto" width="100%" controls autoplay loop muted>
		  <source src="./video/sawyer_push_baseline.mp4" type="video/mp4">
		</video>
		<div class="method-name">SAC</div>
	</div>
	<div class="w3-col s5 w3-center">
		<video height="auto" width="100%" controls autoplay loop muted>
		  <source src="./video/sawyer_push_mopa.mp4" type="video/mp4">
		</video>
		<div class="method-name">MoPA-SAC</div>
	</div>
	<div class="w3-col s1 w3-center">
	</div>
</div>
<span class="env-name"><b>Sawyer Lift</b></span>
- Sawyer arm needs to find a path to get inside a box, grasp a can and take it out from the box. 
<div class="w3-row-padding">
	<div class="w3-col s1 w3-center">
	</div>
	<div class="w3-col s5 w3-center">
		<video height="auto" width="100%" controls autoplay loop muted>
		  <source src="./video/sawyer_lift_baseline.mp4" type="video/mp4">
		</video>
		<div class="method-name">SAC</div>
	</div>
	<div class="w3-col s5 w3-center">
		<video height="auto" width="100%" controls autoplay loop muted>
		  <source src="./video/sawyer_lift_mopa.mp4" type="video/mp4">
		</video>
		<div class="method-name">MoPA-SAC</div>
	</div>
	<div class="w3-col s1 w3-center">
	</div>
</div>
<span class="env-name"><b>Sawyer Assembly</b></span>
- Sawyer arm with an attached table leg needs to avoid other legs to reach a hole of the table, and insert the pole to assemble the table.
<div class="w3-row-padding">
	<div class="w3-col s1 w3-center">
	</div>
	<div class="w3-col s5 w3-center">
		<video height="auto" width="100%" controls autoplay loop muted>
		  <source src="./video/sawyer_assembly_baseline.mp4" type="video/mp4">
		</video>
		<div class="method-name">SAC</div>
	</div>
	<div class="w3-col s5 w3-center">
		<video height="auto" width="100%" controls autoplay loop muted>
		  <source src="./video/sawyer_assembly_mopa.mp4" type="video/mp4">
		</video>
		<div class="method-name">MoPA-SAC</div>
	</div>
	<div class="w3-col s1 w3-center">
	</div>
</div>

----

## Quantitative results

<!-- ![Success Rate](./img/result.png "") -->

<div class="w3-row-padding">
    <div "w3-col s12 w3-center">
        <img src="./img/result.png" alt="Success Rate"/>
    </div>
</div>

----

## Citation
```
@inproceedings{lee2020learning,
  title={Motion Planner Augmented Reinforcement Learning for Obstructed Environments},
  author={Jun Yamada and Youngwoon Lee and Gautam Salhotra and Karl Pertsch and Max Pflueger and Gaurav S. Sukhatme and Joseph J. Lim and Peter Englert},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=ryxB2lBtvH}
}
```
