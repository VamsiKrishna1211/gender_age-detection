import tensorflow as tf


graph_def_file = "tensorflow_model/constant_graph_weights.pb"
input_arrays = ["the_input"]
output_arrays = ["output_node0", "output_node1"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
