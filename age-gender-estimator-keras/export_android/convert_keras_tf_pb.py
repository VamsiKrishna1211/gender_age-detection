# this file is based on https://github.com/amir-abdi/keras_to_tensorflow/

import os
import os.path as osp

from keras import backend as K

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

from models.mobile_net import MobileNetDeepEstimator

model_file = '../checkpoints/weights.62-1.97.hdf5'
print(os.path.exists(model_file))


num_output = 2
write_graph_def_ascii_flag = True
prefix_output_node_names_of_final_network = 'output_node'
output_graph_name = 'constant_graph_weights.pb'

output_fld = "./tensorflow_model"
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)


K.set_learning_phase(0)

num_neu = 21
alpha = 1
input_shape = (224, 224, 3)
net_model = MobileNetDeepEstimator(input_shape[0], alpha, num_neu, weights=None)()
net_model.load_weights(model_file)

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)


# WHY [optional] write graph definition in asci??

sess = K.get_session()
if write_graph_def_ascii_flag:
    f = 'only_the_graph_def.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
    print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))

# convert variables to constants and save

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))


##### safety check graph
pb_exported_file = osp.join(output_fld, output_graph_name)

g = tf.GraphDef()
g.ParseFromString(open(pb_exported_file, "rb").read())
print([n for n in g.node if n.name.find("input") != -1]) # same for output or any other node you want to make sure is ok
print([n for n in g.node if n.name.find("out") != -1]) # same for output or any other node you want to make sure is ok

ops = set([n.op for n in g.node])
print(ops)



