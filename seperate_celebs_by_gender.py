import numpy as np
from random import shuffle
import os
import sys

def create_single(list_of_names, write_file):
    f = open(write_file, 'w')
    for i in xrange(len(list_of_names)):
        f.write(list_of_names[i] + '\n')
    f.close()

if __name__ == "__main__":
    f = open("../list_attr_celeba.txt", "r")
    count = 0
    male_young = []
    female_young = []
    male_old = []   
    female_old = []
    for line in f:
        if count < 2:
            count = count + 1
            continue
        tokens = line.strip().split()
        img_name = tokens[0]
        Blurry_flag = tokens[11]
        Male_flag = tokens[21]
        Young_flag = tokens[40]

        count = count + 1
        if Blurry_flag == "-1":
            if Male_flag == "1":
                if Young_flag == "1":
                    male_young.append(img_name)
                else:
                    male_old.append(img_name)
            else:
                if Young_flag == "1":
                    female_young.append(img_name)
                else:
                    female_old.append(img_name)
    f.close()

    print(len(male_young), len(female_young), len(male_old), len(female_old))

    db_len = 30000
    db_len_test = 2000

    shuffle(male_young)
    shuffle(female_young)
    shuffle(male_old)
    shuffle(female_old)

    fy = female_young[0:db_len]
    my = male_young[0:db_len]

    female_all = female_young[db_len:] + female_old
    shuffle(female_all)
    fa = female_all[0:db_len]

    male_all = male_young[db_len:] + male_old
    shuffle(male_all)
    ma = male_all[0:db_len]

    ft = female_all[db_len:db_len+db_len_test]
    mt = male_all[db_len:db_len+db_len_test]

    create_single(fy, 'dataset/female_young.txt')
    create_single(my, 'dataset/male_young.txt')
    create_single(fa, 'dataset/female_young_old.txt')
    create_single(ma, 'dataset/male_young_old.txt')
    create_single(ft, 'dataset/female_young_old_test.txt')
    create_single(mt, 'dataset/male_young_old_test.txt')
