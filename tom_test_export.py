import tensorflow as tf
import scipy.misc
import model
import cv2
# from subprocess import call

# Yitao-TLS-Begin
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS
# Yitao-TLS-End




sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

i = 0
# while(cv2.waitKey(10) != ord('q')):
full_image = scipy.misc.imread("driving_dataset/" + str(200) + ".jpg", mode="RGB")
image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
# call("clear")
print("Predicted steering angle: " + str(degrees) + " degrees")
# cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
# #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
# #and the predicted angle
# smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
# M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
# dst = cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow("steering wheel", dst)
# i += 1

# cv2.destroyAllWindows()


# Yitao-TLS-Begin
export_path_base = "nvidia_autopilot"
export_path = os.path.join(
            compat.as_bytes(export_path_base),
            compat.as_bytes(str(FLAGS.model_version)))
print 'Exporting trained model to', export_path
builder = saved_model_builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(model.x)
tensor_info_prob = tf.saved_model.utils.build_tensor_info(model.keep_prob)
tensor_info_y = tf.saved_model.utils.build_tensor_info(model.y)

prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x, 'keep_prob': tensor_info_prob},
            outputs={'scores': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature,
            },
            legacy_init_op=legacy_init_op)

builder.save()

print('Done exporting!')
# Yitao-TLS-End