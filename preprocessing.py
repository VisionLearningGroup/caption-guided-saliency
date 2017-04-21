'''
This script process images one at a time. Please, modify to add batch processing support
'''
from os import path, makedirs, stat
import shutil
import subprocess
import glob
import tarfile
from tqdm import tqdm

import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image

import os
import re
import argparse
from cfg import msr_vtt_cfg, flickr_cfg


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', "./DATA/feature_extractors/InceptionV3/", """Path to classify_image_graph_def.pb""")
MODEL_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def maybe_download_and_extract():
    # Download and extract model tar file
    dest_directory = FLAGS.model_dir
    if not path.exists(dest_directory):
        makedirs(dest_directory)
        filename = MODEL_URL.split('/')[-1]
        filepath = path.join(dest_directory, filename)
        if not path.exists(filepath):
            print "Downloading model... "
            filepath, _ = urllib.request.urlretrieve(MODEL_URL, filepath)
            statinfo = stat(filepath)
            print "Succesfully downloaded", filename, statinfo.st_size, "bytes."
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print dest_directory + " exists! Skipping download!"

def create_graph():
    # Creates a graph from saved GraphDef file and returns a saver
    if not path.exists(path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')):
        print "Graph definition " + path.join(FLAGS.model_dir, 'classify_image_graph_def.pb') + " not found"
        return None
    else:
        with tf.gfile.FastGFile(path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        return True

def preprocess_image(img):
    '''
    The shorter side is resized to 256 pixels and center crop 224x244 is taken.
    It follows the same procedure in "Show, Attend and Tell" and "Attention Correctness in Neural Image Captioning".
    Later we resize the cropped image to 299x299 for Inception_V3 compatibility.
    '''
    shorter_side = min(img.shape[:2])
    yy = int((img.shape[0] - shorter_side) / 2)
    xx = int((img.shape[1] - shorter_side) / 2)
    shorter_side_crop = img[yy: yy + shorter_side, xx: xx + shorter_side]
    resized_crop_img = np.asarray(Image.fromarray(shorter_side_crop).resize((256, 256), Image.ANTIALIAS), np.uint8)
    resized_crop_img = resized_crop_img[int((256 - 224) / 2):int((256 - 224) / 2 + 224), int((256 - 224) / 2):int((256 - 224) / 2 + 224)]
    return Image.fromarray(resized_crop_img).resize((299, 299), Image.ANTIALIAS)


def extract_features_flickr30k(cfg):
    with tf.Session() as sess:
        graph_ops = create_graph()
        mixed_10_tensor = sess.graph.get_tensor_by_name('mixed_10/join:0')
        #get list of frames
        dataset = list(set([
            item.split("#")[0] for item in open(cfg.annotations_path).read().split("\n")[:-1]
            ]))
        if not path.exists(cfg.path_to_descriptors):
            makedirs(cfg.path_to_descriptors)
        for image_name in tqdm(dataset):
            image = np.asarray(Image.open(path.join(cfg.path_to_images, image_name)),
                               np.uint8)
            preprocessed_image = np.expand_dims(np.asarray(preprocess_image(image), np.uint8),
                                                axis=0)
            descriptors = sess.run(mixed_10_tensor, {'ResizeBilinear:0':  preprocessed_image})
            np.save(cfg.path_to_descriptors + image_name.split(".")[0]  + cfg.descriptor_suffix,
                    np.squeeze(descriptors))
            
def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        video_id = video.split("/")[-1].split(".")[0]
        if os.path.exists(dst):
            print " cleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = [ "ffmpeg",
        '-y', # (optional) overwrite output file if it exists
        '-i',  video, # input file
        '-vf', "scale=400:300", # input file
        '-qscale:v', "2", #quality for JPEG 
        '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)
        
            
def extract_features_msr_vtt(cfg, video_path, feat_path):
    '''
    for every video:
    1. extract all frames
    2. extract 26 Inception_V3 pool_3 descriptors from uniformly sampled frames 
    3. delete extracted frames
    '''
    with tf.Session() as sess:
        graph_ops = create_graph()
        pool_3_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        if not path.exists(feat_path):
            makedirs(feat_path)
        
        video_list = glob.glob(path.join(video_path, '*.mp4'))
        for video in tqdm(video_list):
            video_id = video.split("/")[-1].split(".")[0]
            dst = video_id
            extract_frames(video, dst)
            
            image_list = sorted(glob.glob(path.join(dst,  '*.jpg')))
            samples = np.round(np.linspace(0, len(image_list) - 1, cfg.n_frame_step))
            image_list = [image_list[int(sample)] for sample in samples]
            
            mixed_10_activations = np.zeros((len(image_list), cfg.dim_image),dtype=np.float32)
            for iImg in range(len(image_list)):
                img = Image.open(image_list[iImg])
                mixed_10_activations[iImg] = sess.run(pool_3_tensor,{'DecodeJpeg:0': img})
                
            # Save the inception features
            outfile = path.join(feat_path, video_id + '_incp_v3.npy')
            np.save(outfile, mixed_10_activations)
            #cleanup
            shutil.rmtree(dst)
            
def main(args):
    # Download Inception_V3 model def
    maybe_download_and_extract()
    if args.dataset == "MSR-VTT":
        cfg = msr_vtt_cfg()
        
        print "Frame extraction/feature pooling for TrainValVideo"
        #extract_features trainval
        extract_features_msr_vtt(cfg,
                                 cfg.path_to_trainval_video,
                                 cfg.path_to_trainval_descriptors)
        
        print "Frame extraction/feature pooling for TestVideo"
        #extract_features test
        extract_features_msr_vtt(cfg,
                                 cfg.path_to_test_video,
                                 cfg.path_to_test_descriptors)
        
    elif args.dataset == "Flickr30k":
        cfg = flickr_cfg()
        extract_features_flickr30k(cfg)
    else:
        print "Unknown dataset"
        exit(1)
    
    print "Done."    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script extract Inception_V3 features for \
                     MSR-VTT and Flickr30k datasets'
        )
    
    parser.add_argument("--dataset", dest='dataset', type=str, required=True,
                        help='Specify the one from {Flickr30k, MSR-VTT}')
    parser.add_argument("--gpu", dest='gpu', type=str, required=False,
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    main(args)        
    