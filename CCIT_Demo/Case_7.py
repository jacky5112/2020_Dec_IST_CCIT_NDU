import numpy as np
from PIL import Image
import tensorflow as tf
import re

#ref: https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/tutorials/image/imagenet/classify_image.py
class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = 'models/imagenet_2012_challenge_label_map_proto.pbtxt'
    if not uid_lookup_path:
      uid_lookup_path = 'models/imagenet_synset_to_human_label_map.txt'
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

session=tf.Session()


adv = tf.get_variable(name="adv", shape=[1,100,100,3], dtype=tf.float32, initializer=tf.zeros_initializer)


#x = tf.placeholder(tf.float32, shape=[1,100,100,3])
target = tf.placeholder(tf.int32)
#assign_op=tf.assign(adv, x)

def create_graph(dirname):
    with tf.gfile.FastGFile(dirname, 'rb') as f:
        graph_def = session.graph_def
        graph_def.ParseFromString(f.read())

        _ = tf.import_graph_def(graph_def, name='adv',
                                input_map={"ExpandDims:0":adv} )


create_graph("models/classify_image_graph_def.pb")


session.run(tf.global_variables_initializer())
tensorlist=[n.name for n in session.graph_def.node]

#print(tensorlist)


softmax_tensor = session.graph.get_tensor_by_name('adv_1/softmax:0')
#input_tensor=session.graph.get_tensor_by_name('ExpandDims:0')
logits_tensor=session.graph.get_tensor_by_name('adv_1/softmax/logits:0')


#imagename="panda.jpg"
imagename="imagen/n07768694_513_pomegranate.jpg"

image=np.array(Image.open(imagename).convert('RGB').resize((100, 100), Image.BILINEAR)).astype(np.float32)
#[100,100,3]->[1,100,100,3]
image=np.expand_dims(image, axis=0)

predictions = session.run(softmax_tensor,
                           {adv: image})
predictions = np.squeeze(predictions)

# Creates node ID --> English string lookup.
node_lookup = NodeLookup()

#top 3
top_k = predictions.argsort()[-3:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print('%s (score = %.5f)(id = %d)' % (human_string, score,node_id))

epochs=500
lr=0.1
target_label=123

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_tensor, labels=[target])
#optimizer = tf.train.GradientDescentOptimizer(lr)
optimizer = tf.train.AdamOptimizer(lr)
train_step=optimizer.minimize(loss=cross_entropy,var_list=[adv])


session.run(tf.global_variables_initializer())


session.run(tf.assign(adv, image))

for epoch in range(epochs):
    
    loss,_,adv_img,predictions=session.run([cross_entropy,train_step,adv,softmax_tensor],{target:target_label})
    
    predictions = np.squeeze(predictions)
    label=np.argmax(predictions)
    
    print("epoch={} loss={} label={}".format(epoch,loss,label))
    

    if label == target_label:
        top_k = predictions.argsort()[-3:][::-1]
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)(id = %d)' % (human_string, score,node_id))
        break