# -- coding: utf-8 --
from nupic.algorithms.spatial_pooler import SpatialPooler as SP
import numpy as np
import csv
from nupic.algorithms.knn_classifier import KNNClassifier
import datetime
start = datetime.datetime.now()
sp = SP(inputDimensions=(1, 784),

        columnDimensions=(1, 2048),###此处是空间池细胞柱的维度，应当是大于inputDimensions

        potentialRadius=784,

        potentialPct=0.85,

        globalInhibition=True,

        localAreaDensity=-1,

        numActiveColumnsPerInhArea=240,

        stimulusThreshold=0,

        synPermInactiveDec=0,

        synPermActiveInc=0,

        synPermConnected=0.2,

        minPctOverlapDutyCycle=0.001,

        dutyCyclePeriod=1000,

        boostStrength=0.0,

        seed=1956)

classifier = KNNClassifier(k=5, distanceNorm=2)
with open('data/h_bit_all_train.csv', 'r') as train_file:
   reader = csv.reader(train_file)
   for count, record in enumerate(reader):
      train = np.array(record[1:785])
      column = np.zeros((2048))

      sp.compute(train, False, column)

      classifier.learn(column, record[0])

train_file.close()
with open('data/h_bit_all_test.csv', 'r') as test_file:
   reader_test = csv.reader(test_file)
   r= csv.writer(open('result.csv', 'w'))

   ans = 0.0
   cla_result = []
   real_record = []
   for test_count,test_record in enumerate(reader_test):
      test = np.array(test_record[1:785])
      column = np.zeros((2048))  ####此处是空间池的细胞柱的维度，与上述的参数中columnDimensions，应当是保持一致

      sp.compute(test,False,column)
      winner = classifier.infer(column)[0]
      cla_result.append(winner)
      real_record.append(test_record[0])
      if winner == int(test_record[0]):
         ans =ans + 1
   r.writerow(cla_result)
   r.writerow(real_record)
   print int(ans)
   ans_per = ans / 100
   print "正确率：", ans_per, "%"
   # r.writerow(ans_per)
test_file.close()
end = datetime.datetime.now()
print (end-start).total_seconds()
