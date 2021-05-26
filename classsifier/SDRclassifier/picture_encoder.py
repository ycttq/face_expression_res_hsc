#coding: utf_8
from nupic.encoders.base import Encoder
import numpy
# from scipy.linalg import hilbert
#
# k = hilbert(3)
# print k
class picture_encoder(Encoder):
  @classmethod
  def encodeIntoArray(self, inputData, output):

    # bit = []
    # for i in range(0, len(inputData)):
    #   if inputData[i] == 0:
    #     for num in range(0, 8):
    #       bit.append(inputData[i])
    #   else:
    #     str = '{:08b}'.format(inputData[i])
    #     for num in range(0, 8):
    #       bit.append(int(str[num]))
    # output_bit_data = numpy.array(bit)
    # output[0:6272] = output_bit_data
    output[0:6272] = inputData
  @classmethod
  def getBucketIndices(self, inputdata):
    """ See method description in base.py """
    loc_bucket = 0
    for i in range(len(inputdata)):
      if inputdata[i] ==  1:
        loc_bucket = loc_bucket +i
        print loc_bucket
    return [loc_bucket]