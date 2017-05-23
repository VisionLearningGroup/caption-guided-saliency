'''
A short script to train and evaluate s2vt model on Flickr30k and
MSR-VTT datasets.

Usage:
            python run_s2vt.py --dataset [MSR-VTT|Flickr30k] --train
            python run_s2vt.py --dataset [MSR-VTT|Flickr30k] --test --checkpoint model_num 
'''

from cfg import *
import os

import pandas as pd
import numpy as np
import pickle as pkl
import json
from tqdm import tqdm
import argparse

from pandas.io.json import json_normalize
from cocoeval import COCOScorer, suppress_stdout_stderr
import sys

from s2vt_model import *

cfg = None


def get_flickr30k_data(cfg):
    #using the provided splits
    train_split = set(map(lambda x: x.split(".")[0], open(cfg.train_file).read().splitlines()))
    val_split = set(map(lambda x: x.split(".")[0], open(cfg.val_file).read().splitlines()))
    test_split = set(map(lambda x: x.split(".")[0], open(cfg.test_file).read().splitlines()))
    
    data = [{"video_id": item.split(".")[0], "sentence_id": item.split("#")[1].split("\t")[0], "caption":item.split("\t")[1]}
            for item in open(cfg.annotations_path).read().splitlines()]
    
    sentences = json_normalize(data)
    sentences['video_path'] = sentences['video_id'].map(lambda x: os.path.join(cfg.path_to_descriptors, x + cfg.descriptor_suffix + ".npy"))
    
    train_imgs = sentences.loc[sentences["video_id"].isin(train_split)]
    train_imgs.reset_index()
    
    val_imgs = sentences.loc[sentences["video_id"].isin(val_split)]
    val_imgs.reset_index()
    
    test_imgs = sentences.loc[sentences["video_id"].isin(test_split)]
    test_imgs.reset_index()
    
    return train_imgs, val_imgs, test_imgs

def get_msr_vtt_data(cfg):
    #trainval data
    with open(cfg.trainval_annotations) as data_file:    
        data = json.load(data_file)
    
    sentences = json_normalize(data['sentences'])
    videos = json_normalize(data['videos'])
    train_vids = sentences.loc[sentences["video_id"].isin(videos[videos['split'] == "train"]["video_id"])]
    val_vids = sentences.loc[sentences["video_id"].isin(videos[videos['split'] == "validate"]["video_id"])]
    train_vids['video_path'] = train_vids['video_id'].map(lambda x: os.path.join(cfg.path_to_trainval_descriptors, x + "_incp_v3.npy"))  
    val_vids['video_path'] = val_vids['video_id'].map(lambda x: os.path.join(cfg.path_to_trainval_descriptors, x + "_incp_v3.npy"))
    
    #test data
    with open(cfg.test_annotations) as data_file:    
        data = json.load(data_file)
    sentences = json_normalize(data['sentences'])
    videos = json_normalize(data['videos'])
    test_vids = sentences.loc[sentences["video_id"].isin(videos[videos['split'] == "test"]["video_id"])]
    test_vids['video_path'] = test_vids['video_id'].map(lambda x: os.path.join(cfg.path_to_test_descriptors, x + "_incp_v3.npy"))
    
    return train_vids, val_vids, test_vids
    

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def output_progress(current, total, loss):
    bar_length = 20
    progress = current/float(total)
    sys.stdout.write('\r')
    sys.stdout.write(("[%-" + str(bar_length) + "s] %d/%d") % ('='* int(bar_length * progress) + ">", current, total) + ", avg_loss=" + str(loss))
    sys.stdout.flush()

#populate feature dictionary
#unroll features for LSTM encoding
feature_dict = {} 
def load_flickr30k_features(vid):
    if vid in feature_dict:
        return feature_dict[vid]
    else:
        temp_array = np.load(vid)
        temp_array[1::2][:] = temp_array[1::2][:, ::-1][:]
        if cfg.use_hard_cache:
            feature_dict[vid] = temp_array.reshape(cfg.n_frame_step, -1)
            return feature_dict[vid]
        else:
            return temp_array.reshape(cfg.n_frame_step, -1)

def load_msr_vtt_features(vid):
    return np.load(vid)
            

def get_validation_loss(sess, current_val_data, wordtoix, tf_loss, tf_video, tf_caption, tf_caption_mask):
    val_data = current_val_data
    val_captions = val_data['caption'].values
    val_captions = map(lambda x: x.replace('.', ''), val_captions)
    val_captions = map(lambda x: x.replace(',', ''), val_captions)
    
    combine_features = load_flickr30k_features if cfg.id == "Flickr30k" else load_msr_vtt_features 
    
    loss_on_validation = []
    for start,end in zip(
                range(0, len(val_data), cfg.batch_size),
                range(cfg.batch_size, len(val_data)+1, cfg.batch_size)): #during every epoch we are discarding incomplete batch in the end
            
        current_batch = val_data[start:end]
        current_videos = current_batch['video_path'].values

        current_feats = np.zeros((cfg.batch_size, cfg.n_frame_step, cfg.dim_image))
        current_feats_vals = map(lambda vid: combine_features(vid), current_videos) 
            
        for ind,feat in enumerate(current_feats_vals):
            current_feats[ind][:len(current_feats_vals[ind])] = feat 
            

        current_captions = current_batch['caption'].values
        current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:cfg.n_lstm_step - 1]
                                                   if word in wordtoix],
                                      current_captions)
        
        current_caption_matrix = np.zeros((cfg.batch_size, cfg.n_lstm_step))
        current_caption_masks = np.zeros((cfg.batch_size, cfg.n_lstm_step))
        for ind, row in enumerate(current_caption_masks):
            valid_length = len(current_caption_ind[ind])
            row[:valid_length] = 1
            current_caption_matrix[ind, :valid_length] = current_caption_ind[ind]
        
        loss_val = sess.run(tf_loss,
                feed_dict={
                    tf_video: current_feats,
                    tf_caption: current_caption_matrix,
                    tf_caption_mask: current_caption_masks
                    })
        loss_on_validation.append(loss_val)
    return np.mean(loss_on_validation)
            
        
def train():
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    
    print cfg.model_path
    f = open(cfg.model_path + "loss", "a", 1)
    f.write("Checkpoint\tTrain loss\tValidation loss\n")
    
    if cfg.id == "Flickr30k":
        train_data, val_data, _ = get_flickr30k_data(cfg)
    elif cfg.id == "MSR-VTT":
        train_data, val_data, _ = get_msr_vtt_data(cfg)
    
    #FIXME add validation data vocabulary
    captions = train_data['caption'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=cfg.word_count_threshold)
    
    combine_features = load_flickr30k_features if cfg.id == "Flickr30k" else load_msr_vtt_features

    np.save(cfg.vocab_path + 'ixtoword', ixtoword)
    with open(cfg.vocab_path + 'wordtoix.pkl', 'wb') as outfile:
        pkl.dump(wordtoix, outfile)
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    
    with tf.variable_scope(tf.get_variable_scope()):
        model_train = s2vt(dim_image=cfg.dim_image,
                           n_words=len(ixtoword),
                           dim_hidden=cfg.dim_hidden,
                           batch_size=cfg.batch_size,
                           n_frame_steps=cfg.n_frame_step,
                           n_lstm_steps=cfg.n_lstm_step,
                           dim_word_emb = cfg.dim_word_emb,
                           cell_clip = cfg.cell_clip,
                           forget_bias = cfg.forget_bias,
                           input_keep_prob = cfg.input_keep_prob,
                           output_keep_prob = cfg.output_keep_prob,
                           bias_init_vector=bias_init_vector)
    
        tf_loss, tf_video, tf_caption, tf_caption_mask, _ = model_train.build_model("training")

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        train_op = tf.train.AdamOptimizer(cfg.learning_rate).minimize(tf_loss)
    
    saver = tf.train.Saver(max_to_keep=cfg.max_to_keep)
    sess.run(tf.global_variables_initializer())
    
    model_counter = 0
    val_loss = None
    for epoch in range(cfg.n_epochs):
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]
        
        current_train_data = train_data
        
        total_loss = 0
        saving_schedule = []
        loss_accumulator = []
        
        step_size =  (int(len(current_train_data) * cfg.save_every_n_epoch) // cfg.batch_size ) * cfg.batch_size
        saving_schedule = range(0, len(current_train_data) - step_size, step_size)
        print saving_schedule
        
        for start,end in zip(
                range(0, len(current_train_data), cfg.batch_size),
                range(cfg.batch_size, len(current_train_data)+1, cfg.batch_size)):   
            current_batch = current_train_data[start:end]                           
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((cfg.batch_size, cfg.n_frame_step, cfg.dim_image))
            current_feats_vals = map(lambda vid: combine_features(vid), current_videos)
            
            
            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat 
                
            current_captions = current_batch['caption'].values
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:cfg.n_lstm_step - 1]
                                                   if word in wordtoix],
                                      current_captions)
            
            current_caption_matrix = np.zeros((cfg.batch_size, cfg.n_lstm_step))
            current_caption_masks = np.zeros((cfg.batch_size, cfg.n_lstm_step))
            for ind, row in enumerate(current_caption_masks):
                valid_length = len(current_caption_ind[ind])
                row[:valid_length+1] = 1                                                #forces to predict <EOS> = 0
                current_caption_matrix[ind, :valid_length] = current_caption_ind[ind]
 
            _, train_loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })
            total_loss += train_loss
            loss_accumulator.append(train_loss)
                
            output_progress(end, len(current_train_data), train_loss)
            
            if start in saving_schedule:
                print start
                train_loss = np.mean(loss_accumulator[-5:])
                val_loss = get_validation_loss(sess, 
                                               val_data.groupby('video_id').apply(lambda x: x.iloc[np.random.choice(len(x))]),
                                               wordtoix, tf_loss, tf_video,
                                               tf_caption, tf_caption_mask)
                f.write(str(model_counter) + "\t" +  str(train_loss) +"\t" + str(val_loss) + "\n")
                sys.stdout.flush()
                saver.save(sess, os.path.join(cfg.model_path, 'model'), global_step=model_counter)
                model_counter+=1
                
        output_progress(end, len(current_train_data), np.mean(loss_accumulator[-5:]))
        print " Done. Validation loss = " + str(val_loss) 
        
        

def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    non_ascii_count = 0
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        try:
            row[0].encode('ascii', 'ignore').decode('ascii')
        except UnicodeDecodeError:
            non_ascii_count+=1
            continue
        if row[1] in gts:
            gts[row[1]].append({u'image_id': row[1], u'cap_id': len(gts[row[1]]), u'caption':row[0].encode('ascii', 'ignore').decode('ascii')})
        else:
            gts[row[1]] = []
            gts[row[1]].append({u'image_id': row[1], u'cap_id': len(gts[row[1]]), u'caption':row[0].encode('ascii', 'ignore').decode('ascii')})
    if non_ascii_count:
        print "=" * 20 + "\n" + "non-ascii: " + str(non_ascii_count) + "\n" + "=" * 20
    return gts


def test(saved_model=''):
    scorer = COCOScorer()
    ixtoword = pd.Series(np.load(cfg.vocab_path + 'ixtoword.npy').tolist())
    combine_features = load_flickr30k_features if cfg.id == "Flickr30k" else load_msr_vtt_features
    
    model = s2vt(dim_image=cfg.dim_image,
                 n_words=len(ixtoword),
                 dim_hidden=cfg.dim_hidden,
                 batch_size=cfg.batch_size,
                 n_frame_steps=cfg.n_frame_step,
                 n_lstm_steps=cfg.n_lstm_step,
                 dim_word_emb = cfg.dim_word_emb,
                 cell_clip = cfg.cell_clip,
                 forget_bias = cfg.forget_bias,
                 input_keep_prob = cfg.input_keep_prob,
                 output_keep_prob = cfg.output_keep_prob,
                 bias_init_vector=None)
    
    _, video_tf, caption_tf, _, _ = model.build_model("inference")
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    saver.restore(session, saved_model)
        
    if cfg.id == "Flickr30k":
        _, _, test_data = get_flickr30k_data(cfg)
    elif cfg.id == "MSR-VTT":
        _, _, test_data = get_msr_vtt_data(cfg)
    
    splits = []
    
    splits.append((test_data['video_path'].unique(), test_data))
    results = []
    for split, gt_dataframe in splits:
        gts = convert_data_to_coco_scorer_format(gt_dataframe)
        samples = {}
        for start,end in zip(
                    range(0, len(split), cfg.batch_size),
                    range(cfg.batch_size, len(split) + cfg.batch_size, cfg.batch_size)):
        
            current_batch = split[start:end]
            current_feats = np.zeros((cfg.batch_size, cfg.n_frame_step, cfg.dim_image))
            current_feats_vals = [combine_features(vid) for vid in current_batch] 
            
            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
            
            generated_word_index = session.run(caption_tf, feed_dict={video_tf:current_feats})
            generated_word_index = np.asarray(generated_word_index).transpose()
            periods = np.argmax(generated_word_index == 0, axis=1) + 1
            periods[periods == 0] = cfg.n_lstm_step      #take the whole sequence if a period was not produced
            for i in range(len(current_batch)):
                generated_sentence = ' '.join(ixtoword[generated_word_index[i, :periods[i]-1]])
                video_id = current_batch[i].split("/")[-1].split("_")[0] #+ ".jpg"
                samples[video_id] = [{u'image_id': video_id, u'caption': generated_sentence}]
                
        with suppress_stdout_stderr():
            valid_score = scorer.score(gts, samples, samples.keys())
        results.append(valid_score)
        print valid_score
    
    print len(samples)            
    if not os.path.exists(cfg.results_path):
        os.makedirs(cfg.results_path)
    
    with open(cfg.results_path + "scores.txt", 'a') as scores_table:
            scores_table.write(json.dumps(results[0]) + "\n") 
    with open(cfg.results_path + saved_model.split("/")[-1] + ".json", 'w') as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score}, prediction_results)


def main(args):
    global cfg
    if args.dataset == "Flickr30k":
        cfg = flickr_cfg()
    elif args.dataset == "MSR-VTT":
        cfg = msr_vtt_cfg()
    else:
        print "Unknown dataset"
        exit(1)
    
    if args.train_stage:
        train()
    else:
        test(saved_model=cfg.model_path + 'model-' + str(args.checkpoint))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train a model for movie description')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', dest='train_stage', action='store_true',  
                        help='Training')
    group.add_argument('--test', dest='train_stage', action='store_false', 
                        help='Testing')
    parser.add_argument('--checkpoint', dest='checkpoint', type = int, default = -1,
                        help='Provide a number of the saved model to run testing only on one snapshot')
    parser.add_argument("--dataset", dest='dataset', type=str,
                        help='Specify one from {Flickr30k, MSR-VTT}')
    parser.add_argument("--gpu", dest='gpu', type=str, required=False,
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    if not args.dataset:
        parser.print_help()
        exit(1)

    if not args.train_stage:
        if args.checkpoint is None:
            parser.print_help()
            exit(1)
    
    main(args)
    