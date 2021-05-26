# -- coding: utf-8 --

from nupic.algorithms.spatial_pooler import SpatialPooler as SP
from nupic.algorithms.temporal_memory import TemporalMemory as TM
import numpy
import csv
from nupic.algorithms.sdr_classifier import SDRClassifier
# from picture_encoder import *
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
sclarencoder = RandomDistributedScalarEncoder(0.88)

sp = SP(inputDimensions= (1, 6272),
        columnDimensions= (1, 6272),
        potentialRadius= 6272,
        potentialPct= 0.85,
        globalInhibition= True,
        localAreaDensity= -1,
        numActiveColumnsPerInhArea= 240,
        stimulusThreshold= 0,
        synPermInactiveDec= 0,
        synPermActiveInc= 0,
        synPermConnected= 0.2,
        minPctOverlapDutyCycle= 0.001,
        dutyCyclePeriod= 1000,
        boostStrength= 0.0,
        seed= 1956)
tm = TM(
    columnDimensions= (1,6772),
    cellsPerColumn= 32,
    activationThreshold= 16,
    initialPermanence= 0.21,
    connectedPermanence= 0.2,
    minThreshold= 12,
    maxNewSynapseCount= 20,
    permanenceIncrement= 0.1,
    permanenceDecrement= 0.1,
    predictedSegmentDecrement=0.0,
    maxSegmentsPerCell= 128,
    maxSynapsesPerSegment= 32,
    seed= 1960
  )
CLA = SDRClassifier(steps=[0],
                    alpha=0.001,
                    actValueAlpha=0.3,
                    verbosity=0)
# output_file = open('result.csv', 'w')
with open('data/h_all_bit_train.csv', 'r') as train_file:
   reader = csv.reader(train_file)
   # csvWriter = csv.writer(output_file)
   # csvWriter.writerow(["real", "oneStep","confidence"])
   for train_count, train_record in enumerate(reader):
      train = numpy.array(train_record[1:6273])
      train_activecolumn = numpy.zeros((6272))
      sp.compute(train, False,train_activecolumn)
      train_activeColumnIndices = numpy.nonzero(train_activecolumn)[0]
      # tm.compute(train_activeColumnIndices, learn=True)
      # train_activaeCell = tm.getActiveCells()
      # bucketIdx = picture_encoder.getBucketIndices(record[1:6273])[0]
      train_bucketIdx = sclarencoder.getBucketIndices(int(train_record[0]))[0]
      # print bucketIdx , record[0]
      ClA_RESULTS = CLA.compute(recordNum=train_count,
                                patternNZ=train_activeColumnIndices,
                                classification={
                                          "bucketIdx": train_bucketIdx, ###获取输入数据的bucket区域，以进行区别分类
                                          "actValue": train_record[0]
                                   },
                                learn=True,
                                infer=False)
      # print (ClA_RESULTS['actualValues'][0])  ###list
      # print (ClA_RESULTS[0].tolist()[0])  ####array

      # csvWriter.writerow([train_record[0], ClA_RESULTS['actualValues'][0],ClA_RESULTS[0].tolist()[0] * 100])
      # print ClA_RESULTS[1]
      # print ClA_RESULTS['actualValues'] ###分类结果
      #
      # print ClA_RESULTS[1]  ###判定概率
      # break
      print train_count
Ture_counts = 0.0
output_file = open('result.csv', 'w')
with open('data/h_all_bit_test.csv', 'r') as test_file:
   reader_test = csv.reader(test_file)
   csvWriter = csv.writer(output_file)
   csvWriter.writerow(["real", "Max_possible_values"])
   for test_count,test_record in enumerate(reader_test):
      test = numpy.array(test_record[1:6273])
      test_activecolumn = numpy.zeros((6272))
      sp.compute(test,False,test_activecolumn)
      test_activeColumnIndices = numpy.nonzero(test_activecolumn)[0]
      # tm.compute(test_activeColumnIndices, learn=True)
      # test_activaeCell = tm.getActiveCells()
      test_bucketIdx = sclarencoder.getBucketIndices(int(train_record[0]))[0]
      CLA_infer_result = CLA.infer(patternNZ=test_activeColumnIndices,actValueList=test_record[0])
      Max_possible_indexs = [i for i , x in enumerate(CLA_infer_result[0].tolist()) if x == max(CLA_infer_result[0].tolist())]
      Max_possible_values = []
      for index in Max_possible_indexs:
          Max_possible_values.append(CLA_infer_result['actualValues'][index])
          csvWriter.writerow([test_record[0], Max_possible_values])
          if test_record[0] in Max_possible_values:
              Ture_counts = Ture_counts + 1
          del Max_possible_indexs[:]
      print test_count
      # print test_record[0],CLA_infer_result
print Ture_counts / 100

####本例子中虽然是设置了SDR分类器的步长 0 ，但是模型的Nontemparlclassiar 应该在此处怎么设置，是否是关闭TM的学习功能
#### 上述是一个疑问点，并且实验结果有待验证。



