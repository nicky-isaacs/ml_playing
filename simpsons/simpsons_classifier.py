import numpy as np
import tensorflow as tf
import csv
from PIL import Image

IMAGE_HEIGHT_PIXLES = 300
IMAGE_WIDTH_PIXELS = 200

class BoundingBox:
    def __init__(self, upper_left_corner, lower_right_corner):
        self.upper_left_corner = upper_left_corner
        self.lower_right_corner = lower_right_corner

class Label
    def __init__(self, name, index, total_label_count):
        self.name = name
        self.index = index
        self.total_label_count = total_label_count

class CroppedAndResizedSimpsonImage:
    def __init__(self, image, label):
        self.image = image
        self.label = label
    
    # Creates a tensor which is a vector of all the flattened RGB values in the image
    def to_tensor():
        # return a tensor of the image in shape [200*300*3]
        image_data=[] 
        for pixel in self.image.getdata:
            r, g, b = pixel
            image_data += [r,g,b]
        return tf.constant(image_data, tf.float64)

    def label_to_tensor():
        zeros = np.zeros(label.total_label_count)
        zeros[label.index]=1
        return zeros
        
class SimpsonsCSVEntry:
    def __init__(self, absolute_file_path, bounding_box, label):
        self.absolute_file_path = absolute_file_path
        self.bounding_box = bounding_box
        self.label = label

def crop_and_resize(img, bounding_box):
    return img\
        .crop((bounding_box.upper_left_corner[0], bounding_box.upper_left_corner[1], bounding_box.lower_right_corner[0], bounding_box.lower_right_corner[1]))\
        .resize((IMAGE_HEIGHT_PIXELS, IMAGE_WIDTH_PIXELS), Image.ANTIALIAS)

def simpons_csv_to_cropped_or_none(simpsons_csv):
    image = Image.open(simpsons_csv.absolute_file_path)
    try:
        image.verify()
        return CroppedAndResizedSimpsonsImage(
            crop_and_resize(image, bounding_box),
            simpsons_csv.label)
    except:
        return None

# Format: filepath,x1,y1,x2,y2,character 
def read_csv(path):
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            yield SimpsonsCSVEntry(row[0], BoundingBox((int(row[1]), int(row[2])), (int(row[3]), int(row[4]))), row[5])

def sort_and_uniq_labels(names):
    as_set=list(set(name))
    as_set.sort()
    return as_set

if __name__ == "__main__":
    csv_entries=[]
    label_strings=[]
    for csv_entry in read_csv(sys.argv[1]):
       entry = simpons_csv_to_cropped_or_none(csv_entry) 
       csv_entries += entry
       label_strings += entry.label
    labels = sort_and_uniq_labels(label_strings)  
    total_labels=size(labels)
    
    # check the flag for location of the .csv file
    # read in the csv file, parsed into SimpsonsCSVEntry objects
    # filter out the bad ones
    # apply bounding box to the images
    # resize to AxB where A and B are constants
    # turn these things into tensors
