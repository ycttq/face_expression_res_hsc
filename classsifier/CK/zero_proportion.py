import csv
with open('data/face_all_bit_test.csv', 'r') as test_file:
   reader = csv.reader(test_file)
   count = 0.0
   for  i , record in enumerate(reader):
        print len(record)
   #      for num in list(record):
   #          if num == '1':
   #              count = count +1
   #          else:
   #              pass
   #      print count
   #      count = 0.0
   # print i