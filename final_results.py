import os.path
from scipy import ndimage
import numpy as np
from skimage.io import imsave

f_file = open('test_good_female.txt', 'r')
m_file = open('test_good_male.txt', 'r')

f_list = []
m_list = []

for line in f_file:
    f_list.append(line.strip())
for line in m_file:
    m_list.append(line.strip())

img_root = '../img_align_celeba/high/'
offspring_root = 'test_good_dump/mean/'

serial = 1
b_size = 10

output = []
for _ in xrange(2000):

    f_name = f_list[int(np.random.random() * len(f_list))]
    m_name = m_list[int(np.random.random() * len(m_list))]

    if os.path.isfile(offspring_root + f_name + '_' + m_name + '_f_0.jpg') == False:
        print(offspring_root + f_name + '_' + m_name + '_f_0.jpg')
        continue

    f_in_img = ndimage.imread(img_root + f_name + '.jpg', mode='RGB')
    m_in_img = ndimage.imread(img_root + m_name + '.jpg', mode='RGB')

    op = []
    op.append((f_in_img, f_name + '.jpg'))
    op.append((m_in_img, m_name + '.jpg'))
    for ci in xrange(5):
        offspring_f_name = f_name + '_' + m_name + '_f_' + str(ci) + '.jpg' 
        offspring_m_name = f_name + '_' + m_name + '_m_' + str(ci) + '.jpg' 
        offspring_f_img = ndimage.imread(offspring_root + offspring_f_name, mode='RGB')
        offspring_m_img = ndimage.imread(offspring_root + offspring_m_name, mode='RGB')
        op.append(offspring_f_img)
        op.append(offspring_m_img)
    output.append(op)

    if len(output) == b_size:

        '''
        # All splits.
        n_row = b_size
        n_col = 12

        img_row = 218
        img_col = 178
        draw_string = ''    
        image = np.zeros((img_row*n_row, img_col*n_col + 350, 3), dtype=np.uint8)
        for i in xrange(n_row):
            image[i*img_row:(i+1)*img_row, img_col*0:img_col*1, :] = output[i][0][0]
            image[i*img_row:(i+1)*img_row, img_col*1:img_col*2, :] = output[i][1][0]
            for ci in xrange(5):
                image[i*img_row:(i+1)*img_row, img_col*2+img_col*ci:img_col*2+img_col*(ci+1), :] = output[i][2+ci*2]
            for ci in xrange(5):
                image[i*img_row:(i+1)*img_row, img_col*7+img_col*ci:img_col*7+img_col*(ci+1), :] = output[i][3+ci*2]

            draw_string = draw_string + 'text 2200,' + str(100 + i * img_row) + ' \'' + output[i][0][1] + '\' '
            draw_string = draw_string + 'text 2200,' + str(150 + i * img_row) + ' \'' + output[i][1][1] + '\' '
            
        imname = 'final_results/all_split/sample_' + str(serial) + '.jpg'
        imsave(imname, image)
        os.system('convert ' + imname + ' -stroke white -fill white -pointsize 36 -font Rachana-Regular -draw \"' + draw_string + '\" ' + imname)
        '''

        # Only 100_0 splits.
        n_row = b_size
        n_col = 6 

        img_row = 218
        img_col = 178
        draw_string = ''    
        image = np.zeros((img_row*n_row, img_col*n_col + 350, 3), dtype=np.uint8)
        for i in xrange(n_row):
            image[i*img_row:(i+1)*img_row, img_col*0:img_col*1, :] = output[i][0][0]
            image[i*img_row:(i+1)*img_row, img_col*1:img_col*2, :] = output[i][2]
            image[i*img_row:(i+1)*img_row, img_col*2:img_col*3, :] = output[i][3]
            image[i*img_row:(i+1)*img_row, img_col*3:img_col*4, :] = output[i][1][0]
            image[i*img_row:(i+1)*img_row, img_col*4:img_col*5, :] = output[i][10]
            image[i*img_row:(i+1)*img_row, img_col*5:img_col*6, :] = output[i][11]
     
            draw_string = draw_string + 'text 1132,' + str(100 + i * img_row) + ' \'' + output[i][0][1] + '\' '
            draw_string = draw_string + 'text 1132,' + str(150 + i * img_row) + ' \'' + output[i][1][1] + '\' '
            
        imname = 'final_results/only_100_0_split/sample_' + str(serial) + '.jpg'
        imsave(imname, image)
        os.system('convert ' + imname + ' -stroke white -fill white -pointsize 36 -font Rachana-Regular -draw \"' + draw_string + '\" ' + imname)

        serial = serial + 1
        output = []


