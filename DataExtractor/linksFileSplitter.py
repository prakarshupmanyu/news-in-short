import csv
import os
import numpy

file = '/home/melvin/Documents/USC/news-in-short-data/urls/' +  "lesechos-fr-spider-links_politique.csv"
links = []
with open(file, "r") as f:
    links = f.readlines()

print(links)
x = range(25)
l = numpy.array_split(numpy.array(x), 6)