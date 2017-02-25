import csv
import os
import numpy

file = '/home/sarthak/Mydata/Projects/silicon-beach-data/urls/' +  "lemonde-fr-Spider-links_politique.csv"
links = []
with open(file, "r") as f:
    links = f.readlines()

print(links)
x = range(25)
l = numpy.array_split(numpy.array(x), 6)