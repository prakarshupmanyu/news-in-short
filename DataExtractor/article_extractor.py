# encoding: utf-8

from __future__ import unicode_literals
import csv
import urllib2
import re
import os, sys
from time import time
import goose
from goose import Goose
import boto
import smart_open
import boto.s3.connection
from multiprocessing import Pool
from uuid import getnode as get_mac

print "-----------------------------------------------------------------------"
print "-----------------------------------------------------------------------\n\n"
try:
    # Put access_key and secret_key here
    access_key = sys.argv[1] 
    secret_key =  sys.argv[2]
except IndexError:
    print "Please add your S3 access key and secret key as arguments: python article_extractor <access key> <secret key> <start row>"
    sys.exit(0)

try:
    start_value = sys.argv[3]
except IndexError:
    print "Starting from row :: 0"
    start_value = 0


conn = boto.connect_s3(access_key, secret_key)

bucket = conn.get_bucket('news-in-short', validate=False)

files = []
my_mac = get_mac()
user_macs = [145376947886732, 146162447908330, 202866753072730, 48853196067632]

for key in bucket.list():
    if "links_politique.csv" in key.name:
        files.append(key.name)

files.sort()
dirs = [files[i] for i in range(len(files)) if i%len(user_macs) == user_macs.index(my_mac)]
path = 'https://s3-us-west-1.amazonaws.com/news-in-short/data/'

csvLinkFiles_path = path
print(dirs)

for file in dirs:
    try:
        start_time = time()
        newspaper = file.split("-")[0]
        category = file.split("_")[1].split(".")[0]
        
        print(category)
        print(newspaper)

        linksList = []

        read_key = bucket.get_key(file)
        
        with smart_open.smart_open(read_key) as fin:
            for line in fin:
                linksList.append(line)

        """

        with open(csvFilename) as f:
            content = f.readlines()
            for link in content:
                if(len(link) != 0):
                    linksList.append(link)
        """

        linksList = linksList[start_value:]
        
        number_of_threads = 3
        number_of_articles_each = 1 + len(linksList) / number_of_threads
        print "No of articles to be extracted : " + str(len(linksList))
        print "No of threads currently being executed : " + str(number_of_threads)
        print "No of articles in each thread : " + str(number_of_articles_each)
        print "-----------------------------------------------------------------------"
        print "-----------------------------------------------------------------------\n\n"

        def get_articles(start_value):
            article_count = 0
            incompatible_count = 0
            end_value = start_value + number_of_articles_each
            new_file = file.replace("data","art").replace("links", "data_"+str(start_value) + "-" + str(end_value))
            write_key = bucket.new_key(new_file)
            
            for link in linksList[start_value:end_value]:
                try:
                    g = goose.Goose({'target_language': 'fr'})
                    article = g.extract(url=link)
                    mainarticle = article.cleaned_text
                    title = article.title

                    response = urllib2.urlopen(link)
                    html = response.read()

                    #print("Title :" , title)
                    #print("Category :", category)
                    print link
                    #print("newspaper name :", newspaper)
                    #print(mainarticle)
                    #print("\n###############\n")

                    datarow = []

                    datarow.append(title.encode('UTF-8'))
                    datarow.append(link)
                    datarow.append(category.encode('UTF-8'))
                    datarow.append(newspaper.encode('UTF-8'))
                    datarow.append(mainarticle.encode('UTF-8'))

                    with open(new_file, 'a') as mycsvfile:
                        thedatawriter = csv.writer(mycsvfile)
                        thedatawriter.writerow(datarow)
                        article_count+=1
                        #print("row written :", article_count)
                    
                except Exception as e:
                    print("Sorry incompatible article...  reason  : ",e)
                    incompatible_count+=1

            output_file = open(new_file,'rb')
            write_key.set_contents_from_file(output_file)
            print(new_file + " written to S3")
            return [article_count, incompatible_count]


        # get compatible and imcompatible articles
        com_in = Pool(number_of_threads).map(get_articles,list(xrange(start_value,len(linksList),number_of_articles_each)))

        end_time = time()
    except Exception as error:
        print(error)