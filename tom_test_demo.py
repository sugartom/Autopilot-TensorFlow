from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import scipy.misc
import numpy as np

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

# import tensorflow as tf
import scipy.misc
# import model
import cv2
from subprocess import call

import time

# sess = tf.InteractiveSession()
# saver = tf.train.Saver()
# saver.restore(sess, "save/model.ckpt")

host, port = "localhost:9000".split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'nvidia_autopilot'
request.model_spec.signature_name = 'predict_images'

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

i = 0

startTime = time.time()

while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

    # degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    keep_prob = 1.0
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, dtype = np.float32, shape=[1, 66, 200, 3]))
    request.inputs['keep_prob'].CopyFrom(
        tf.contrib.util.make_tensor_proto(keep_prob))
    result = stub.Predict(request, 10.0)

    degrees = float(result.outputs["scores"].float_val[0]) * 180.0 / scipy.pi


    call("clear")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

endTime = time.time()

print("Average frame rate: %s" % str(1 / ((endTime - startTime) / i)))

cv2.destroyAllWindows()
