# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for Importance Weighted Actor-Learner Architectures.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.

Note that this is a copy of the code previously published by Lasse Espeholt
under an Apache license at:
https://github.com/deepmind/scalable_agent
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from streetlearn.python.agents.city_nav_agent import CityNavAgent
from streetlearn.python.agents.goal_nav_agent import GoalNavAgent
from streetlearn.python.agents.locale_pathway import LocalePathway
from streetlearn.python.agents.plain_agent import PlainAgent
