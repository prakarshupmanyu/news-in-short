import csv
import sys
import glob


csv.field_size_limit(sys.maxsize)

count = 0

for art_file in glob.glob("*.csv"):
    new_count = 0
    with open(art_file, "r") as f:
        for row in csv.reader(f):
            new_count += 1

    print(art_file, new_count)
    count += new_count
    if count > 10000:
        break

print("Total", count)
