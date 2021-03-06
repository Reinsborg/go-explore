#!/bin/sh
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# OBS
# This file is not in its original state as released by Uber Technologies, Inc.
# Alteration made by:
# Jeppe Reinsborg, 3 June 2019
#

./phase1.sh \
    --high_score_weight=10.0 \
    --horiz_weight=0.3 \
    --vert_weight=0.1 \
    --batch_size 1000 \
    --explore_steps=100 \
    --max_hours=480 \
    --max_compute_steps=40000000 \
    --remember_rooms \
    --expl='ppo' \
    --log_path='log/'
    --save_model
#--pictures \
	#--pp \
	#--ip \
	#--kpp \
	#--kip \
	#--resolution=16

