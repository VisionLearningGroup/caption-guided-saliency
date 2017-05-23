'''
Saliency visualization for test video from MSR-VTT dataset
'''
import os
from os import path, makedirs
import glob
from tqdm import tqdm
import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image
import argparse
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import subprocess
from scipy.ndimage.filters import convolve1d, gaussian_filter

from cfg import *
from s2vt_model import *
from preprocessing import extract_frames

FLAGS = tf.app.flags.FLAGS

###############################CONFIG
data_path = "./DATA/MSR-VTT/"
#tf.app.flags.DEFINE_string('model_dir', "./DATA/" + "feature_extractors/InceptionV3/", """Path to classify_image_graph_def.pb""")
frames_path = data_path + "test_frames/"
descriptors_save_path = data_path + "full_descriptors/"
path_to_save_figures = "./output_samples/"
SCALE = 300                                 #defines the size of single image in a grid         
###############################CONFIG


        
def extract_all_features(video_id):
    #using Inception_V3
    def create_graph():
            # Creates a graph from saved GraphDef file and returns a saver
            if not path.exists(path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')):
                print("Graph definition " + path.join(FLAGS.model_dir, 'classify_image_graph_def.pb') + " not found")
                exit(1)
            else:
                with tf.gfile.FastGFile(path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(graph_def, name='')
    
    dst = path.join(frames_path, video_id)
    extract_frames(path.join(data_path, "TestVideo/", video_id + ".mp4"), dst)
    
    with tf.Session() as sess:
        create_graph()
        video_frames = sorted(glob.glob(path.join(dst,'*.jpg')))
        
        if not path.exists(descriptors_save_path):
            makedirs(descriptors_save_path)
    
        mixed_10_tensor = sess.graph.get_tensor_by_name('mixed_10/join:0')
        video_descriptors = []
        for image_path in tqdm(video_frames):
            with open(image_path, 'rb') as image_data:
                descriptors = sess.run(mixed_10_tensor, {'DecodeJpeg/contents:0': image_data.read()})
            video_descriptors.append(np.squeeze(descriptors))
        video_descriptors = np.asarray(video_descriptors, dtype=np.float32)
        np.save( path.join(descriptors_save_path, video_id  + '_incp_v3.npy'), video_descriptors)
    #cleanup
    tf.reset_default_graph()
    return video_descriptors
    
    
class Saliency_Generator():
    def __init__(self, cfg, checkpoint):
        '''
        Creates the model for per frame saliency prediction where all spatial features
        forms a batch with input sequence length 1
        '''
        ixtoword = pd.Series(np.load(cfg.vocab_path + 'ixtoword.npy').tolist())
        wordtoix = pkl.load(open(cfg.vocab_path + 'wordtoix.pkl'))
        
        
        #create the model for saliency estimation
        model_train = s2vt(dim_image=cfg.dim_image,
                       n_words=len(ixtoword),
                       dim_hidden=cfg.dim_hidden,
                       batch_size=cfg.n_frame_step,
                       n_frame_steps=1,
                       n_lstm_steps=cfg.n_lstm_step,
                       dim_word_emb = cfg.dim_word_emb,
                       cell_clip = cfg.cell_clip,
                       forget_bias = cfg.forget_bias,
                       input_keep_prob = cfg.input_keep_prob,
                       output_keep_prob = cfg.output_keep_prob,
                       bias_init_vector=None)
        

        _, tf_video, tf_caption, tf_caption_mask, tf_frame_saliency_maps = model_train.build_model("saliency")
        
        session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        saver = tf.train.Saver()
        saver.restore(session, path.join(cfg.experiment,  "model-" + str(checkpoint)))
        
        self.ixtoword = ixtoword
        self.wordtoix = wordtoix 
        self.tf_video = tf_video
        self.tf_caption = tf_caption
        self.tf_caption_mask = tf_caption_mask
        self.tf_frame_saliency_maps = tf_frame_saliency_maps
        self.cfg = cfg
        self.session = session

        
    def get_saliency_map_single_frame(self, input_dict):
        raw_saliency_maps = \
        self.session.run(self.tf_frame_saliency_maps,
                         input_dict) ##
        return raw_saliency_maps
        
        
    def get_saliency_map_video(self, video_features, input_sentence):
        '''
        video_features:    n_frames x 8 x 8 x 2048
        counterintuitive because batch_size = n_frame_step previously
        '''
        video_saliency_maps = []
        w_ind = [self.wordtoix[word] for word in input_sentence.lower().split(' ')[:self.cfg.n_lstm_step - 1]]
        valid_length = len(w_ind)
        assert len(w_ind) == len(input_sentence.split(' '))  
        w_ind += [0] * (self.cfg.n_lstm_step - len(w_ind))
        w_ind = np.asarray(w_ind, dtype=np.int32).reshape(1, -1)
        batched_w_ind = w_ind.repeat(self.cfg.n_frame_step, axis=0) # np.zeros((self.cfg.batch_size, self.cfg.n_lstm_step))
        w_masks = np.zeros((self.cfg.n_frame_step, self.cfg.n_lstm_step))
        w_masks[:, :valid_length + 1] = 1
        
        #get raw values for saliency maps using get_attention_map_single_frame
        for idx in tqdm(range(len(video_features))):
            input_dict = {self.tf_video         : video_features[idx].reshape(self.cfg.n_frame_step, 1, self.cfg.dim_image),
                          self.tf_caption       : batched_w_ind, 
                          self.tf_caption_mask  : w_masks}
            frame_saliency_maps = self.get_saliency_map_single_frame(input_dict)
            video_saliency_maps.append(frame_saliency_maps[:valid_length])
        return np.asarray(video_saliency_maps)
     
    
def saliency_normalization(saliency_maps):
    n_frames, n_words = saliency_maps.shape[:2]
    
    def sentence_normalization(saliency_maps):
        minimum = np.amin(saliency_maps, axis=(1, 2))
        maximum = np.amax(saliency_maps, axis=(1, 2))
        delta = maximum - minimum
        for idx in range(n_frames):
            saliency_maps[idx, :, :] = (saliency_maps[idx, :, :] - minimum[idx]) / delta[idx]
        return saliency_maps

    def time_normalization(saliency_maps):
        '''
        saliency of every word is scaled over time to [0, 1] 
        '''
        minimum = np.amin(saliency_maps, axis=(0, 2))
        maximum = np.amax(saliency_maps, axis=(0, 2))
        delta = maximum - minimum
        for idx in range(n_words):
            saliency_maps[:, idx, :] = (saliency_maps[:, idx, :] - minimum[idx]) / delta[idx]
        
        #thresholding
        mean = np.mean(saliency_maps, axis=(0, 2))
        for idx in range(n_words):
            saliency_maps[:, idx, :] = np.clip((saliency_maps[:, idx, :] - mean[idx]), 0, 1)        
        return saliency_maps
    
    saliency_maps = time_normalization(saliency_maps)  #works like thresholding on mean
    saliency_maps = sentence_normalization(saliency_maps)
    
    return saliency_maps

def get_superimposed_frame(video_id, frame_filename, saliency_frame, sentence):
    from matplotlib.font_manager import FontProperties
    font0 = FontProperties()
    font0.set_family("sans-serif")
    
    def generate_saliency(spatial_attention, image_size, norm = False):
        minimum = np.amin(spatial_attention)
        maximum = np.amax(spatial_attention)
        
        spatial_attention = np.pad(spatial_attention, pad_width = ((1, 1), (1, 1)), 
               mode = 'constant', constant_values=np.amin(spatial_attention))
        
        saliency_image = Image.fromarray(np.uint8((spatial_attention) * float(255)), 'L').resize(image_size, Image.BICUBIC)
        saliency_image = saliency_image.resize((int(image_size[0] * 1.2), int(image_size[1] * 1.2)), Image.BICUBIC)
        saliency_image = saliency_image.crop((int(image_size[0] * 0.1 ), int(image_size[1] * 0.1 ), int(image_size[0] * 1.1), int(image_size[0] * 1.1) ))
        
        return saliency_image 
    
    
    original_image = Image.open(frame_filename).resize((SCALE, SCALE),
                                                        Image.ANTIALIAS)
    n_words = saliency_frame.shape[0]
    
    w = np.floor(np.sqrt(n_words))
    h = np.ceil(np.float32(n_words) / w )
    figw, figh = int(h * 3), int(w * 3)
    f = plt.figure(figsize=(figw, figh), facecolor = "black", dpi = 150)
    
    for word_idx in range(saliency_frame.shape[0]):
        plt.subplot(w, h, word_idx+1)    
        plt.imshow(original_image)
        saliency = generate_saliency(saliency_frame[word_idx].reshape(8, 8),
                                     (SCALE, SCALE), norm = False)
        saliency = np.asarray(saliency) / 255.
        plt.imshow(saliency, vmin=0.0, vmax=1.0, alpha = 0.5)
        
        fontsize = 12 + (h - 2) * 2 
        plt.text(6, 18, sentence[word_idx], fontproperties = font0,
                 color = "black", backgroundcolor='white', fontsize=fontsize)
        
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off',
                        bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')
    
    bezel_thickness = 0.02
    plt.tight_layout(pad = bezel_thickness, w_pad=bezel_thickness, h_pad=bezel_thickness)
    plt.subplots_adjust(hspace = bezel_thickness , # ,
                        wspace = bezel_thickness )
    
    plt.savefig(path.join(path_to_save_figures, video_id, frame_filename.split("/")[-1] + ".png"),
                bbox_inches='tight',
                facecolor=f.get_facecolor(),
                dpi=90,
                edgecolor='none')
    plt.close()

def temporal_feature_smoothing(video_features, kernel):
    #simple 1d convolution assuming that input is time x words x descriptors
    return convolve1d(video_features, weights = kernel, axis = 0)

def gaussian_filter_3d(saliency_maps, sigma = None):
    if not sigma:
        sigma = [2, 1, 1]
    
    for word in range(saliency_maps.shape[1]):
        smoothed = gaussian_filter(saliency_maps[:, word, :].reshape(-1, 8, 8), sigma = sigma)
        saliency_maps[:, word, :] = smoothed.reshape(-1, 8 * 8)
    
    return saliency_maps     
    
def create_video_example(video_id, cfg, checkpoint, input_sentence = "a man is driving a car"):
    #load/extract features
    existing_file = path.join(descriptors_save_path, video_id + cfg.descriptor_suffix + ".npy")
    print "Frame feature extraction..."
    if path.exists(existing_file):
        video_features = np.load(existing_file)
    else:
        video_features = extract_all_features(video_id)
    video_features = temporal_feature_smoothing(video_features,
                                                np.ones(3, dtype=np.float32)/float(3.))
    
    print "Saliency computation..."
    cfg.n_frame_step = 64                                       #considering all locations as separate steps
    saliency_generator = Saliency_Generator(cfg, checkpoint)
    #expecting to get here full list of corresponding saliency maps 
    saliency_maps = saliency_generator.get_saliency_map_video(video_features, input_sentence) 
    
    #postprocessing for better visualization
    saliency_maps = saliency_normalization(saliency_maps)
    
    #smoothing of saliency values
    saliency_maps = gaussian_filter_3d(saliency_maps, sigma = [2, 1, 1])
     
    #perform overlay
    video_frames = sorted(glob.glob(path.join(frames_path,video_id,'*.jpg')))
    assert len(video_frames) == saliency_maps.shape[0]
    if not path.exists(path.join(path_to_save_figures, video_id)):
        makedirs(path.join(path_to_save_figures, video_id))
    
    #it is not necessary to save all frames using pyplot since it is too slow for rendering,
    #the same thing with PIL and direct pipe to ffmpeg would work ~40x faster
    print "Overlaying frames with saliency (pyplot)..."
    for i in tqdm(range(len(video_frames))):
        get_superimposed_frame(video_id, video_frames[i], saliency_maps[i], input_sentence.split(" "))
    
    with open(os.devnull, "w") as ffmpeg_log:
        frames_to_video_command = [ "ffmpeg",
                '-y', # (optional) overwrite output file if it exists
                '-framerate', '8',
                '-start_number', '1',
                '-i', "output_samples/{0}/%06d.jpg.png".format(video_id),
                '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=25,format=yuv420p", 
                '-c:v', 'libx264',
                '-preset', 'veryslow',
                '-profile:v', 'high',
                '-level', '4.0',
                '{0}.mp4'.format(video_id)]
        subprocess.call(frames_to_video_command, stdout=ffmpeg_log, stderr=ffmpeg_log)
    
def main(args):
    if args.dataset == "MSR-VTT":
        cfg = msr_vtt_cfg()
        create_video_example(args.media_id, cfg, args.checkpoint, args.sentence)
    
    elif args.dataset == "Flickr30k":
        #add standard visualization for Flickr30k
        print "Not implemented"
        exit(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to produce visualizations')
    
    parser.add_argument("--dataset", dest='dataset', type=str, required=True,
                        help='Specify the one from {Flickr30k, MSR-VTT}')
    parser.add_argument('--media_id', dest='media_id', type=str, required=True,  
                        help='Put either video_id or image_id from test splits')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, required=True,  
                        help='Specify the number of a checkpoint for the model')
    parser.add_argument('--sentence', dest='sentence', type=str, required=True,  
                        help='Sentence for visualization')
    parser.add_argument("--gpu", dest='gpu', type=str, required=False,
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(args)