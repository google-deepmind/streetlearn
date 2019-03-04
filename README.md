# StreetLearn

## Overview

This repository contains an implementation of the
[**StreetLearn**](http://streetlearn.cc) environment for training navigation
agents as well as code for implementing the agents used in the NeurIPS 2018
paper on
["Learning to Navigate in Cities Without a Map"](http://papers.nips.cc/paper/7509-learning-to-navigate-in-cities-without-a-map).
The StreetLearn environment relies on panorama images from
[Google Street View](https://maps.google.com) and provides an interface for
moving a first-person view agent inside the Street View graph. This is not an
officially supported Google product.

For a detailed description of the architecture please read our paper. Please
cite the paper if you use the code from this repository in your work.

Our paper also provides a detailed description of how to train and implement
navigation agents in the StreetLearn environment by using a TensorFlow
implementation of "Importance Weighted Actor-Learner Architectures", published
in Espeholt, Soyer, Munos et al. (2018) "IMPALA: Scalable Distributed Deep-RL
with Importance Weighted Actor-Learner
Architectures"(https://arxiv.org/abs/1802.01561). The generic agent and trainer
code have been published by Lasse Espeholt under an Apache license at:
[https://github.com/deepmind/scalable_agent](https://github.com/deepmind/scalable_agent).

### Bibtex

```
@inproceedings{mirowski2018learning,
  title={Learning to Navigate in Cities Without a Map},
  author={Mirowski, Piotr and Grimes, Matthew Koichi and Malinowski, Mateusz and Hermann, Karl Moritz and Anderson, Keith and Teplyashin, Denis and Simonyan, Karen and Kavukcuoglu, Koray and Zisserman, Andrew and Hadsell, Raia},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2018}
}
```

### Code structure

This environment code contains:

*   **streetlearn/engine** Our C++ StreetLearn engine for loading, caching and
    serving Google Street View panoramas by projecting them from a
    equirectangular representation to first-person projected view at a given
    yaw, pitch and field of view, and for handling navigation (moving from one
    panorama to another) depending on the city street graph and the current
    orientation.
*   **streetlearn/proto** The message
    [protocol buffer](https://developers.google.com/protocol-buffers/) used to
    store panoramas and street graph.
*   **streetlearn/python/environment** A Python-based interface for calling the
    StreetLearn environment with custom action spaces. Within the Python
    StreetLearn interface, several games are defined in individual files whose
    names end with **_game.py**.
*   **streetlearn/python/ui** A simple interactive **human_agent** and an
    **oracle_agent** and **instruction_following_oracle_agent** for courier and
    instruction-following tasks respectively; all agents are implemented in
    Python using pygame and instantiate the StreetLearn environment on the
    requested map, along with a simple user interface. The interactive
    **human_agent** enables a user to play various games. The **oracle_agent**
    and **instruction_following_oracle_agent** are similar to the human agent
    and automatically navigate towards the goal (courier game) or towards the
    goal via waypoints, following instructions (instruction-following game) and
    they report oracle performance on these tasks.

## Compilation from source

[Bazel](http://bazel.build) is the official build system for StreetLearn. The
build has only been tested running on Ubuntu 18.04.

### Install build prerequisites

```shell
sudo apt-get install autoconf automake libtool curl make g++ unzip virtualenv python-virtualenv cmake subversion pkg-config libpython-dev libcairo2-dev libboost-all-dev python-pip libssl-dev
pip install setuptools
pip install pyparsing
```

### Install Protocol Buffers

For detailed information see:
https://github.com/protocolbuffers/protobuf/blob/master/src/README.md

```shell
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
./autogen.sh
./configure
make -j7
sudo make install
sudo ldconfig
cd python
python setup.py build
sudo python setup.py install
cd ../..
```

### Install CLIF

```shell
git clone https://github.com/google/clif.git
cd clif
./INSTALL.sh
cd ..
```

### Install OpenCV 2.4.13

```shell
wget https://github.com/opencv/opencv/archive/2.4.13.6.zip
unzip 2.4.13.6.zip
cd opencv-2.4.13.6
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j7
sudo make install
sudo ldconfig
cd ../..
```

### Install Python dependencies

```shell
pip install six
pip install absl-py
pip install inflection
pip install wrapt
pip install numpy
pip install dm-sonnet
pip install tensorflow-gpu
pip install tensorflow-probability-gpu
pip install pygame
```

### Install Bazel

[This page](https://bazel.build/) describes how to install the Bazel build and
test tool on your machine.

### Building StreetLearn

Clone this repository and install
[Scalable Agent](https://github.com/deepmind/scalable_agent):

```shell
git clone https://github.com/deepmind/streetlearn.git
cd streetlearn
sh get_scalable_agent.sh
```

To build the StreetLearn engine only:

```shell
export CLIF_PATH=$HOME/opt
bazel build streetlearn:streetlearn_engine_py
```

To build the human agent and the oracle agent in the StreetLearn environment,
with all the dependencies:

```shell
export CLIF_PATH=$HOME/opt
bazel build streetlearn/python/ui:all
```

## Running the StreetLearn human agent

To run the human agent using one of the StreetLearn datasets downloaded and
stored at **dataset_path**:

```shell
bazel run streetlearn/python/ui:human_agent -- --dataset_path=<dataset path>
```

For help with the options of the human_agent:

```shell
bazel run streetlearn/python/ui:human_agent -- --help
```

Similarly, to run the oracle agent on the courier game:

```shell
bazel run streetlearn/python/ui:oracle_agent -- --dataset_path=<dataset path>
```

The human agent and the oracle agent show a **view_image** (on top) and a
**graph_image** (on bottom).

### Actions available to an agent:

*   Rotate left or right in the panorama, by a specified angle (change the yaw
    of the agent). In the human_agent, press **a** or **d**.
*   Rotate up or down in the panorama, by a specified angle (change the pitch of
    the agent). In the human_agent, press **w** or **s**.
*   Move from current panorama A forward to another panorama B if the current
    bearing of the agent from A to B is within a tolerance angle of 30 degrees.
    In the human_agent, press **space**.
*   Zoom in and out in the panorama. In the human_agent, press **i** or **o**.

Additional keys for the human_agent are **escape** and **p** (to print the
current view as a bitmap image).

For training RL agents, action spaces are discretized using integers. For
instance, in our paper, we used 5 actions: (move forward, turn left by 22.5 deg,
turn left by 67.5 deg, turn right by 22.5 deg, turn right by 67.5 deg).

### Navigation Bar

Along the bottom of the **view_image** is the navigation bar which displays a
small circle in any direction in which travel is possible:

*   When within the centre range, they will turn green meaning the user can move
    in this direction.
*   When they are out of this range, they will turn red meaning this is
    inaccessible.
*   When more than one dots are within the centre range, all except the most
    central will turn orange, meaning that there are multiple (forward)
    directions available.

### Stop signs

The graph is constructed by breadth first search to the depth specified by the
graph depth flags. At the maximum depth the graph will suddenly stop, generally
in the middle of a street. Because we are trying to train agents to recognize
streets as navigable, and in order not to confuse the agents, red stop signs are
shown from two panoramas away from any terminal node in the graph.

### Obtaining the StreetLearn dataset

You can request the StreetLearn dataset on the
[StreetLearn project website](https://sites.google.com/view/streetlearn/).

## Using the StreetLearn environment code

The Python StreetLearn environment follows the specifications from
[OpenAI Gym](https://gym.openai.com/docs/). The call to function
**step(action)** returns: * **observation** (tuple of observations requested at
construction), * **reward** (a float with the current reward of the agent), *
**done** (boolean indicating whether the episode has ended) * and **info** (a
dictionary of environment state variables). After creating the environment, it
is initialised by calling function **reset()**. If the flag auto_reset is set to
True at construction, **reset()** will be called automatically every time that
an episode ends.

### Environment Settings

Default environment settings are stored in streetlearn/python/default_config.py.

*   **seed**: Random seed.
*   **width**: Width of the streetview image.
*   **height**: Height of the streetview image.
*   **graph_width**: Width of the map graph image.
*   **graph_height**: Height of the map graph image.
*   **status_height**: Status bar height in pixels.
*   **field_of_view**: Horizontal field of view, in degrees.
*   **min_graph_depth**: Min bound on BFS depth for panos.
*   **max_graph_depth**: Max bound on BFS depth for panos.
*   **max_cache_size**: Pano cache size.
*   **bbox_lat_min**: Minimum value for normalizing the target latitude.
*   **bbox_lat_max**: Maximum value for normalizing the target latitude.
*   **bbox_lng_min**: Minimum value for normalizing the target longitude.
*   **bbox_lng_max**: Maximum value for normalizing the target longitude.
*   **min_radius_meters**: Minimum distance from goal at which reward shaping
    starts in the courier game.
*   **max_radius_meters**: Maximum distance from goal at which reward shaping
    starts in the courier game.
*   **timestamp_start_curriculum**: Integer timestamp (UNIX time) when
    curriculum learning starts, used in the curriculum courier game.
*   **hours_curriculum_part_1**: Number of hours for the first part of
    curriculum training (goal location within minimum distance), used in the
    curriculum courier game.
*   **hours_curriculum_part_2**: Number of hours for the second part of
    curriculum training (goal location annealed further away), used in the
    curriculum courier game.
*   **min_goal_distance_curriculum**: Distance in meters of the goal location at
    the beginning of curriculum learning, used in the curriculum courier game.
*   **max_goal_distance_curriculum**: Distance in meters of the goal location at
    the beginning of curriculum learning, used in the curriculum courier game.
*   **instruction_curriculum_type**: Type of curriculum learning, used in the
    instruction following games.
*   **frame_cap**: Episode frame cap.
*   **full_graph**: Boolean indicating whether to build the entire graph upon
    episode start.
*   **sample_graph_depth**: Boolean indicating whether to sample graph depth
    between min_graph_depth and max_graph_depth.
*   **start_pano**: The pano ID string to start from. The graph will be build
    out from this point.
*   **graph_zoom**: Initial graph zoom. Valid between 1 and 32.
*   **show_shortest_path**: Boolean indicator asking whether the shortest path
    to the goal shall be shown on the graph.
*   **calculate_ground_truth**: Boolean indicator asking whether the ground
    truth direction to the goal should be calculated during the game (useful for
    oracle agents, visualisation and for imitation learning).
*   **neighbor_resolution**: Used to calculate a binary occupancy vector of
    neighbors to the current pano.
*   **color_for_touched_pano**: RGB color for the panos touched by the agent.
*   **color_for_observer**: RGB color for the observer.
*   **color_for_coin**: RGB color for the panos containing coins.
*   **color_for_goal**: RGB color for the goal pano.
*   **color_for_shortest_path**: RGB color for panos on the shortest path to the
    goal.
*   **color_for_waypoint**: RGB color for a waypoint pano.
*   **observations**: Array containing one or more names of the observations
    requested from the environment: ['view_image', 'graph_image', 'yaw',
    'pitch', 'metadata', 'target_metadata', 'latlng', 'target_latlng',
    'latlng_label', 'target_latlng_label', 'yaw_label', 'neighbors',
    'thumbnails', 'instructions', 'ground_truth_direction']
*   **reward_per_coin**: Coin reward for coin game.
*   **reward_at_waypoint**: Waypoint reward for the instruction-following games.
*   **reward_at_goal**: Goal reward for the instruction-following games.
*   **proportion_of_panos_with_coins**: The proportion of panos with coins.
*   **game_name**: Game name, can be: 'coin_game', 'exploration_game',
    'courier_game', 'curriculum_courier_game', 'goal_instruction_game',
    'incremental_instruction_game' and 'step_by_step_instruction_game'.
*   **action_spec**: Either of 'streetlearn_default', 'streetlearn_fast_rotate',
    'streetlearn_tilt'
*   **rotation_speed**: Rotation speed in degrees. Used to create the action
    spec.
*   **auto_reset**: Boolean indicator whether games are reset automatically when
    the max number of frames is achieved.

### Observations

The following observations can be returned by the agent:

*   **view_image**: RGB image for the first-person view image returned from the
    environment and seen by the agent,
*   **graph_image**: RGB image for the top-down street graph image, usually not
    seen by the agent,
*   **yaw**: Scalar value of the yaw angle of the agent, in degrees (zero
    corresponds to North),
*   **pitch**: Scalar value of the pitch angle of the agent, in degrees (zero
    corresponds to horizontal),
*   **metadata**: Message protocol buffer of type Pano with the metadata of the
    current panorama,
*   **target_metadata**: Message protocol buffer of type Pano with the metadata
    of the target/goal panorama,
*   **latlng**: Tuple of lat/lng scalar values for the current position of the
    agent,
*   **target_latlng**: Tuple of lat/lng scalar values for the target/goal
    position,
*   **latlng_label**: Integer discretized value of the current agent position
    using 1024 bins (32 bins for latitude and 32 bins for longitude),
*   **target_latlng_label**: Integer discretized value of the target position
    using 1024 bins (32 bins for latitude and 32 bins for longitude),
*   **yaw_label**: Integer discretized value of the agent yaw using 16 bins,
*   **neighbors**: Vector of immediate neighbor egocentric traversability grid
    around the agent, with 16 bins for the directions around the agent and bin 0
    corresponding to the traversability straight ahead of the agent.
*   **thumbnails**: Array of n+1 RGB images for the first-person view image
    returned from the environment, that should be seen by the agent at specific
    waypoints and goal locations when playing the instruction-following game
    with n instructions,
*   **instructions**: List of n string instructions for the agent at specific
    waypoints and goal locations when playing the instruction-following game
    with n instructions,
*   **ground_truth_direction**: Scalar value of the relative ground truth
    direction to be taken by the agent in order to follow a shortest path to the
    next goal or waypoint. This observation should be requested only for agents
    trained using imitation learning.

### Games

The following games are available in the StreetLearn environment:

*   **coin_game**: invisible coins scattered throughout the map, yielding a
    reward of 1 for each. Once picked up, these rewards do not reappear until
    the end of the episode.
*   **courier_game**: the agent is given a goal destination, specified as
    lat/long pairs. Once the goal is reached (with 100m tolerance), a new goal
    is sampled, until the end of the episode. Rewards at a goal are proportional
    to the number of panoramas on the shortest path from the agent's position
    when it gets the new goal assignment to that goal position. Additional
    reward shaping consists in early rewards when the agent gets within a range
    of 200m of the goal. Additional coins can also be scattered throughout the
    environment. The proportion of coins, the goal radius and the early reward
    radius are parametrizable.
*   **curriculum_courier_game**: same as the courier game, but with a curriculum
    on the difficulty of the task (maximum straight-line distance from the
    agent's position to the goal when it is assigned).
*   **goal_instruction_game** and its variations
    **incremental_instruction_game** and **step_by_step_instruction_game** use
    navigation instructions to direct agents to a goal. Agents are provided with
    a list of instructions as well as thumbnails that guide the agent from its
    starting position to the goal location. In
    **step_by_step_instruction_game**, agents are provided one instruction and
    two thumbnails at a time, in the other game variants the whole list is
    available throughout the whole game. Reward is granted upon reaching the
    goal location (all variants), as well as when hitting individual waypoints
    (**incremental_instruction_game** and **step_by_step_instruction_game**
    only). During training various curriculum strategies are available to the
    agents, and reward shaping can be employed to provide fractional rewards
    when the agent gets within a range of 50m of a waypoint or goal.

## License

The Abseil C++ library is licensed under the terms of the Apache license. See
[LICENSE](LICENSE) for more information.

## Disclaimer

This is not an official Google product.
