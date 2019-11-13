import os
import cv2
import click
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from object_detection.data_decoders import tf_example_decoder
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, ElementTree


def make_elements(dom, root_node, element_map):
    """ add sub-node for root_node """
    node = dom.createElement(element_map.get('name'))
    root_node.appendChild(node)
    if element_map.get('text') != 'None':
        text = dom.createTextNode(str(element_map.get('text')))
        node.appendChild(text)
    if element_map.get('sub') != 'None':
        for item in element_map.get('sub'):
            make_elements(dom, node, item)


def write_labels(img_name, boxes, label_path):
    f = open(label_path + img_name + '.txt', 'w')
    for box in boxes:
        f.write(
            str(0) + ' '
            + str(box[0]) + ' '
            + str(box[1]) + ' '
            + str(box[2]) + ' '
            + str(box[3]) + '\n')
    f.close()


def write_to_voc(image_raw, save_path):
    """ generate annotations from tfrecord only one image infomation """
    image = image_raw['image'][...,::-1] # BGR <=> RGB
    img_name = str(image_raw['source_id'], encoding='utf-8')
    if img_name.split('.')[-1] != 'jpg':
        img_name = img_name + '.jpg'
    img_path = save_path + 'JPEGImages/' + img_name
    cv2.imwrite(img_path, image)
    xml_path = save_path + 'Annotations/' + img_name.rsplit('.', 1)[0] + '.xml'
    img_shape = image_raw['original_image_spatial_shape']
    classes = image_raw['groundtruth_classes']
    boxes = image_raw['groundtruth_boxes']
    
    label_path = save_path + 'labels/'
    write_labels(img_name.rsplit('.', 1)[0], boxes, label_path)
    
    dom = minidom.Document()
    root_node = dom.createElement('annotation')
    dom.appendChild(root_node)
    
    root_common_subelements = [
        {'name': 'folder', 'text': 'person', 'sub': 'None'},
        {'name': 'filename', 'text': img_name, 'sub': 'None'},
        {'name': 'source', 'text': 'None', 'sub': [
            {'name': 'database', 'text': 'The VOC2007 Database', 'sub': 'None'},
            {'name': 'annotation', 'text': 'PASCAL VOC2007', 'sub': 'None'},
            {'name': 'image', 'text': 'default', 'sub': 'None'},
            {'name': 'flickrid', 'text': 'default', 'sub': 'None'},
        ]},
        {'name': 'owner', 'text': 'None', 'sub': [
            {'name': 'flickrid', 'text': 'defalut', 'sub': 'None'},
            {'name': 'name', 'text': 'defalut', 'sub': 'None'},
        ]},
        {'name': 'size', 'text': 'None', 'sub': [
            {'name': 'width', 'text': img_shape[1], 'sub': 'None'},
            {'name': 'height', 'text': img_shape[0], 'sub': 'None'},
            {'name': 'depth', 'text': 3, 'sub': 'None'},
        ]},
        {'name': 'segmented', 'text': 0, 'sub': 'None'},
    ]
    for common_suelement in root_common_subelements:
        make_elements(dom=dom, root_node=root_node, element_map=common_suelement)
        
    for category, box in zip(classes, boxes):
        element_map = {
            'name': 'object', 
            'text': 'None',
            'sub': [
                {'name': 'name', 'text': 'person', 'sub': 'None'},
                {'name': 'pose', 'text': 'Left', 'sub': 'None'},
                {'name': 'truncated', 'text': 1, 'sub': 'None'},
                {'name': 'difficult', 'text': 0, 'sub': 'None'},
                {'name': 'bndbox', 'text': 'None', 'sub': [
                    {'name': 'xmin', 'text': box[0] * img_shape[1], 'sub': 'None'},
                    {'name': 'ymin', 'text': box[1] * img_shape[0], 'sub': 'None'},
                    {'name': 'xmax', 'text': box[2] * img_shape[1], 'sub': 'None'},
                    {'name': 'ymax', 'text': box[3] * img_shape[0], 'sub': 'None'}
                ]}
            ]
        }
        make_elements(dom, root_node, element_map)
    
    with open(xml_path, 'w') as xml:
        dom.writexml(xml, indent='', addindent='\t', newl='\n', encoding='utf-8')
        print('write to {} done'.format(xml_path))


def read_tf_records(tf_record_path):
    """ read tfrecord from record path """
    graph = tf.Graph()
    label_map_proto_file = None
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with graph.as_default():
        decoder = tf_example_decoder.TfExampleDecoder(
            label_map_proto_file=label_map_proto_file
        )

        dataset = tf.data.TFRecordDataset(tf_record_path)
        dataset = dataset.map(decoder.decode)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with tf.train.MonitoredTrainingSession(config=config) as sess:
            while not sess.should_stop():
                example = sess.run(next_element)
                yield example

                
def generate_voc(dataset, voc_folder):
    """  generate voc dataset from tensorflow dataset """
    for idx, example in enumerate(dataset):
        image = example['image']
        detection_dict = dict(
            detection_boxes=example['groundtruth_boxes'],
            detection_classes=example['groundtruth_classes'],
            detection_scores=np.ones(example['groundtruth_classes'].shape)
        )
        write_to_voc(example, voc_folder)
        if idx < 2:
            plt.figure()
            plt.imshow(image)
            plt.show()
        
    print("show ended")


def write_train_txt(voc_folder, annotations_folder, images_folder):
    train_txt = voc_folder + 'ImageSets/Main/train.txt'
    if os.path.exists(train_txt):
        os.remove(train_txt)
    with open(train_txt, 'a') as f:
        for xml_path in os.listdir(annotations_folder):
            f.write(images_folder + xml_path.rsplit('.', 1)[0] + '.jpg' + '\n')
#             f.write(xml_path.rsplit('.', 1)[0] + '\n')
        f.close()

def main():
    tfrecord_location = os.path.abspath('./records/')
    # name = "Person_20191018_10.record"
    name = "train.record"
    # name = "test.record"
    filename = os.path.join(tfrecord_location, name)
    record_exists = os.path.exists(filename)
    record_exists

    voc_folder = "/app/chris/darknet/build-release/voc/VOCdevkit/person/"
    annotations_folder = voc_folder + 'Annotations/'
    images_folder = voc_folder + 'JPEGImages/'
    # dataset = read_tf_records(filename)
    # generate_voc(dataset, voc_folder)
    # write_train_txt()

main()
