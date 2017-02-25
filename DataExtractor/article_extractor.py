# encoding: utf-8

from __future__ import unicode_literals
import csv
import urllib2
import re
import os, sys
from time import time
import goose
from goose import Goose


path = '/home/sarthak/Mydata/Projects/silicon-beach-data/urls/'

csvLinkFiles_path = path
dirs = os.listdir(csvLinkFiles_path)
print(dirs)

for file in dirs:
    try:
        start_time = time()
        csvFilename = csvLinkFiles_path+ file
        newspaper = file.split("-")[0]
        category = file.split("_")[1].split(".")[0]
        print(category)
        print(newspaper)

        linksList = []

        with open(csvFilename) as f:
            content = f.readlines()
            for link in content:
                if(len(link) != 0):
                    linksList.append(link)

        print("no of articles to be extracted : ", len(linksList))


        datafilename = file.replace("links", "data")
        article_count = 0
        incompatible_count = 0
        print(datafilename)

        dataStoragefile = path + datafilename

        for link in linksList:
            try:
                g = goose.Goose({'target_language': 'fr'})
                article = g.extract(url=link)
                mainarticle = article.cleaned_text
                title = article.title

                response = urllib2.urlopen(link)
                html = response.read()


                print("Title :" , title)
                print("Category :", category)
                print("Article Link :", link)
                print("newspaper name :", newspaper)
                print(mainarticle)

                print("\n###############\n")

                datarow = []

                datarow.append(title.encode('UTF-8'))
                datarow.append(link)
                datarow.append(category.encode('UTF-8'))
                datarow.append(newspaper.encode('UTF-8'))
                datarow.append(mainarticle.encode('UTF-8'))


                with open(dataStoragefile, 'a') as mycsvfile:
                    thedatawriter = csv.writer(mycsvfile)
                    thedatawriter.writerow(datarow)
                    article_count+=1
                    print("row written :", article_count)

            except Exception as e:
                print("Sorry incompatible article...  reason  : ",e)
                incompatible_count+=1



        end_time = time()

    except Exception as error:
        print("NA File")

print(path + "..... Done")







