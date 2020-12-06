import csv
import json
import os
import sys
os.chdir('C:\\Users\\Vargha\\Desktop\\Symo\\New folder\\images')

count = 1
for filename in (os.listdir()):
    src = filename
    dst = str('%04d' % count) + ".jpg"
    os.rename(src, dst)
    count += 1



csvFilePath = 'C:\\Users\\Vargha\\Desktop\\Symo\\New folder\\data.csv'
jsonFilePath = 'C:\\Users\\Vargha\\Desktop\\Symo\\New folder\\data.json'
data = []
jsonFile = open(jsonFilePath, 'w')
with open(csvFilePath) as csvFile:
    csvReader = csv.DictReader(csvFile)
    for rows in csvReader:
        data.append(rows)

jsonFile.write(json.dumps(data))










