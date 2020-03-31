#!/usr/bin/env bash

python -m tensorflow.python.tools.optimize_for_inference \
--input tensorflow_model/constant_graph_weights.pb \
--output tensorflow_model/constant_graph_weights_optimized.pb \
--input_names=the_input \
--output_names=output_node0
