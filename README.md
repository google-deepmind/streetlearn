# StreetLearn

## Overview

This repository contains an implementation of the **StreetLearn** environment
for training navigation agents as well as code for implementing the agents used
in the NeurIPS 2018 paper on
["Learning to Navigate in Cities Without a Map"](http://papers.nips.cc/paper/7509-learning-to-navigate-in-cities-without-a-map).
The StreetLearn environment relies on panorama images from
[Google Street View](https://maps.google.com) and provides an interface for
moving a first-person view agent inside the Street View graph. This is not an
officially supported Google product.

For a detailed description of the architecture please read our paper. Please
cite the paper if you use the code from this repository in your work.

### Bibtex

```
@article{mirowski2018learning,
  title={Learning to Navigate in Cities Without a Map},
  author={Mirowski, Piotr and Grimes, Matthew Koichi and Malinowski, Mateusz and Hermann, Karl Moritz and Anderson, Keith and Teplyashin, Denis and Simonyan, Karen and Kavukcuoglu, Koray and Zisserman, Andrew and Hadsell, Raia},
  journal={arXiv preprint arXiv:1804.00168},
  year={2018}
}
```

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
```

## Install CLIF

```shell
git clone https://github.com/google/clif.git
cd clif
./INSTALL.sh
```

## Install OpenCV 2.4.13

```shell
wget https://github.com/opencv/opencv/archive/2.4.13.6.zip
unzip 2.4.13.zip
cd opencv-2.4.13.6
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j7
sudo make install
sudo ldconfig
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

### Build StreetLearn

To build the environment only:

```shell
export CLIF_PATH=$HOME/opt
bazel build streetlearn:streetlearn_engine_py
```

To run the human agent:

```shell
export CLIF_PATH=$HOME/opt
bazel build streetlearn/python/human_agent
bazel run streetlearn/python/human_agent -- --dataset_path=<dataset path>
```

## Environment Settings

Default environment settings are stored in streetlearn/python/default_config.py.

*   **width**: Width of rendered window.
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
*   **frame_cap**: Episode frame cap.
*   **full_graph**: Boolean indicating whether to build the entire graph upon
    episode start.
*   **sample_graph_depth**: Boolean indicating whether to sample graph depth
    between min_graph_depth and max_graph_depth.
*   **start_pano**: The pano ID string to start from. The graph will be build
    out from this point.
*   **graph_zoom**: Initial graph zoom. Valid between 1 and 32.
*   **neighbor_resolution**: Used to calculate a binary occupancy vector of
    neighbors to the current pano.
*   **color_for_observer**: RGB color for the observer.
*   **color_for_coin**: RGB color for the panos containing coins.
*   **color_for_goal**: RGB color for the goal pano.
*   **observations**: Array containing one or more names of the observations
    requested from the environment: ['view_image', 'graph_image', 'yaw',
    'pitch', 'metadata', 'target_metadata', 'latlng', 'target_latlng',
    'yaw_label', 'neighbors']
*   **reward_per_coin**: Coin reward for coin game.
*   **proportion_of_panos_with_coins**: The proportion of panos with coins.
*   **level_name**: Level name, can be: 'coin_game', 'exploration_game'.
*   **action_spec**: Either of 'streetlearn_default', 'streetlearn_fast_rotate',
    'streetlearn_tilt'
*   **rotation_speed**: Rotation speed in degrees. Used to create the action
    spec.

## Observations

The following observations can be returned by the agent:

*   **view_image**: RGB image for the first-person view image returned from the
    environment and seen by the agent,
*   **graph_image**: RGB image for the top-down street graph image, usually not
    seen by the agent,
*   **yaw**: Scalar value of the yaw angle of the agent, in degrees (zero
    corresponds to North),
*   **pitch**: Scalar value of the pitch angle of the agent, in degrees (zero
    corresponds to horizontal),
*   **metadata**: Message proto buffer of type Pano with the metadata of the
    current panorama,
*   **metadata**: Message proto buffer of type Pano with the metadata of the
    target/goal panorama,
*   **latlng**: Tuple of lat/lng scalar values for the current position of the
    agent,
*   **target_latlng**: Tuple of lat/lng scalar values for the target/goal
    position,
*   **yaw_label**: Integer discretized value of the agent yaw using 16 bins,
*   **neighbors**: Vector of immediate neighbor egocentric traversability grid
    around the agent, with 16 bins for the directions around the agent and bin 0
    corresponding to the traversability straight ahead of the agent.

### Navigation Bar

Along the bottom of the UI is the navigation bar which displays a small circle
in any direction in which travel is possible:

*   When within the centre range, they will turn green meaning the user can move
    in this direction.
*   When they are out of this range, they will turn red meaning this is
    inaccessible.
*   When more than one dots are within the centre range, all except the most
    central will turn orange meaning there is a multiple direction choice.

### Stop signs

The graph is constructed by breadth first search to the depth specified by the
graph depth flags. At the maximum depth the graph will suddenly stop, generally
in the middle of a street. Since we are trying to train agents to recognize
streets as nagivable this can be misleading. for this reason no entry signs are
shown from two panos away from any terminal node in the graph.

## License

The Abseil C++ library is licensed under the terms of the Apache license. See
[LICENSE](LICENSE) for more information.

## Disclaimer

This is not an official Google product.
