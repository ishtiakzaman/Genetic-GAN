def analyze_one():
    f = open('mturk_result_1.csv', 'r')
    cnt = -1
    gender_correct = 0 
    similarity_correct = 0
    for line in f:
        if cnt == -1:
            cnt = 0
            continue
        tokens = line.strip().split(',')
        #print(line)
        #print(tokens)
        #print(tokens[29], tokens[29][:-7][-1], tokens[29][:-5][-1])
        #print(tokens[30], tokens[30][:27][-1])
        #print(tokens[31], tokens[31][:55][-1])

        print(tokens[29])
        gender_original = tokens[29][:-7][-1] # f, m
        index_original = int(tokens[29][:-5][-1]) # [0,4]
        if 'label' in tokens[30]:
            gender_mturk = tokens[30][:38][-1] # F, M
        else:
            gender_mturk = tokens[30][:27][-1] # F, M

        if 'label' in tokens[31]:
            index_mturk = tokens[31][:55][-1] # m, f, o (m-more like mother, f-more like father, o-50/50)
            print(tokens[31][:55])
        else:
            index_mturk = tokens[31][:44][-1] # m, f, o (m-more like mother, f-more like father, o-50/50)
            print(tokens[31][:44])

        if (gender_original == 'f' and gender_mturk == 'F') or (gender_original == 'm' and gender_mturk == 'M'):
            gender_correct = gender_correct + 1

        if index_mturk == 'm' and (index_original == 0 or index_original == 1):
            similarity_correct = similarity_correct + 1
        if index_mturk == 'f' and (index_original == 3 or index_original == 4):
            similarity_correct = similarity_correct + 1
        if index_mturk == 'o' and (index_original == 2):
            similarity_correct = similarity_correct + 1
        cnt = cnt + 1

    print(str(gender_correct) + '/' + str(cnt))
    print(str(similarity_correct) + '/' + str(cnt))

def analyze_three():
    f = open('mturk_result_3.csv', 'r')
    cnt = -1
    index_correct = 0 
    for line in f:
        if cnt == -1:
            cnt = 0
            continue
        tokens = line.strip().split(',')
        #print(line)
        #print(tokens)
        #print(tokens[29], tokens[29][:-7][-1], tokens[29][:-5][-1])
        #print(tokens[30], tokens[30][:27][-1])
        #print(tokens[31], tokens[31][:55][-1])

        index_original = int(tokens[29][:-5][-1]) # [0,2]
        #print(tokens[29], index_original)
        #print(tokens[30], tokens[30][:-6][-1])
        index_mturk = int(tokens[30][:-6][-1]) # [1,3]
        if index_original + 1 == index_mturk:
            index_correct = index_correct + 1
    
        cnt = cnt + 1

    print(str(index_correct) + '/' + str(cnt))

def analyze_four():
    f = open('mturk_result_4.csv', 'r')
    cnt = -1
    vote_our = 0 
    for line in f:
        if cnt == -1:
            cnt = 0
            continue
        tokens = line.strip().split(',')
        #print(line)
        #print(tokens[29], tokens[29][:-5][-1])
        #print(tokens[30], tokens[30][:27][-1])
        #print(tokens[31], tokens[31][:55][-1])

        index_our = int(tokens[29][:-5][-1]) # [0,1]
        print(tokens[29], index_our)
        print(tokens[30], tokens[30][:-6][-1])
        index_mturk = tokens[30][:-6][-1] # [1,2]
        if index_mturk == 'e':
            continue
        index_mturk = int(index_mturk)
        if index_our + 1 == index_mturk:
            vote_our = vote_our + 1
    
        cnt = cnt + 1

    print(str(vote_our) + '/' + str(cnt))

analyze_four()
