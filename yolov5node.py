import rospy
import threading
import torch

import numpy as np
import cv2 as cv

from geometry_msgs.msg import PointStamped

from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import Image, RegionOfInterest

class yolov5node(object):
  def __init__(self):
    rospy.logwarn("Initializing yolov5 screw detection node")
    self._last_msg = None
    self._msg_lock = threading.Lock()
    self._model = torch.hub.load('ultralytics/yolov5', 'custom', path='../best.pt')
    self._model.names = ['passive screw', 'screw']

    self._result_pub = rospy.Publisher('~result', Float32MultiArray, queue_size=1)
    self._vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
    self._sub = rospy.Subscriber('/cam_1/color/image_raw', Image, self.callback_image, queue_size=1)

  def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        '''
        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
        else:
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)
        '''
        return cv_img

  def run(self):
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
      if self._msg_lock.acquire(False):
        img_msg = self._last_msg
        self._last_msg = None
        self._msg_lock.release()
      else:
        rate.sleep()
        continue

      if img_msg is not None:
        
        img_yolo = self.convert_to_cv_image(img_msg)

        results = self._model(img_yolo)
        result_msg = self.getResult(results)
        self._result_pub.publish(result_msg)
        
  def getResult(self, results):      
      #filter screw/passive screw by 'class' value, here screw=1
      xyxy_screw = results.xyxy[0][(results.xyxy[0][:, 5] == 1).nonzero().squeeze(1)]
      
      #get center coordinates      
      x_center = (xyxy_screw[:,0] + xyxy_screw[:,2])/2
      y_center = (xyxy_screw[:,1] + xyxy_screw[:,3])/2
      center_coords = np.transpose(np.vstack((x_center.numpy(), y_center.numpy())))
      print('center_coords = ', center_coords)
      
      size1 = center_coords.shape[0]
      size2 = center_coords.shape[1]

      dim1 = MultiArrayDimension(label='layout', size=size1, stride=size1*size2)

      layout= MultiArrayLayout()
      layout.dim.append(dim1)

      msg = Float32MultiArray()
      msg.layout = layout
      msg.data = center_coords.reshape(center_coords.size).tolist()


  def callback_image(self, msg):
    rospy.logdebug("Get an image")
    if self._msg_lock.acquire(False):
        self._last_msg = msg
        self._header = msg.header
        self._msg_lock.release()
 

  '''
  @staticmethod
  def load_param(param, default=None): #?
    new_param = rospy.get_param(param, default)
    rospy.loginfo("[yolov5] %s: %s", param, new_param)
    return new_param
  '''

def main():
  rospy.init_node('yolov5_node', anonymous=True)
  node = yolov5node()
  node.run()

if __name__ == '__main__':
  main()