import imageio
import numpy as np
import h5py
import sys

def read_input_file(input_file):
    file = open(input_file, 'r')
    image_names = []
    for line in file:
        line = line.strip()
        image_names.append(line.encode('utf8'))
    file.close()
    return image_names

def convert_to(image_list, image_root, record_name):
   
    num_examples = len(image_list)    
    print(num_examples, 'total images to process')

    img = imageio.imread(image_root+image_list[0].decode('utf-8'))
    print('Image shape:', img.shape)

    filename = record_name + '.h5'
    print('Writing to', filename)

    file = h5py.File(filename, 'w')
    label_list = np.expand_dims(np.array(image_list), axis=1)
    dataset_labels = file.create_dataset('name', data=label_list, dtype='S80')
    dataset_images = file.create_dataset('image', (num_examples, img.shape[0], img.shape[1], img.shape[2]), compression=None, dtype='i8')

    img_list = None
    batch_size = 64
    for index in range(num_examples):        
        img = imageio.imread(image_root+image_list[index].decode('utf-8'))
        if img.ndim < 3: # bw photo
            img = np.expand_dims(img, axis=2)
            img = np.concatenate((img, img, img), axis=2)
        
        if img_list is None:
            img_list = np.expand_dims(img, axis=0)
        else:
            img = np.expand_dims(img, axis=0)
            img_list = np.concatenate((img_list, img))

        if (index+1) % batch_size == 0:
            dataset_images[index-batch_size+1:index+1] = img_list
            img_list = None
            print(index+1, 'images processed')

    if img_list is not None:
        dataset_images[num_examples-img_list.shape[0]:num_examples] = img_list

    print(num_examples, 'images processed')

if __name__ == '__main__':    

    if len(sys.argv) != 4:
        print('Argument Error, Usage:', sys.argv[0], '<image_root> <input_text_file> <record_name>')
        print('Example:', sys.argv[0], 'images/ train.txt train')
        sys.exit(0)
    
    image_root = sys.argv[1]
    input_file = sys.argv[2]
    record_name = sys.argv[3]

    image_list = read_input_file(input_file)
    convert_to(image_list, image_root, record_name)
