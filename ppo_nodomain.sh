#
# OBS
# This file is not part of the original release by Uber Technologies, Inc.
# Alteration made by:
# Jeppe Reinsborg, 3 June 2019
#

./phase1.sh \
    --state_is_pixels \
    --explorer="ppo" \
    --seen_weight=3.0 \
    --chosen_weight=0.1 \
    --high_score_weight=0.0 \
    --max_hours=120 \
    --max_compute_steps=300000000

