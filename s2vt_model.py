import tensorflow as tf
import numpy as np
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

class s2vt():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_frame_steps,
                 n_lstm_steps, dim_word_emb, cell_clip, forget_bias, input_keep_prob,
                 output_keep_prob, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_frame_steps = n_frame_steps
        self.n_lstm_steps = n_lstm_steps
        self.dim_word_emb = dim_word_emb
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        
        
        self.Wemb = tf.get_variable("Wemb", 
                                    shape = [n_words, dim_word_emb],
                                    initializer = tf.contrib.layers.xavier_initializer())
        
        self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units = dim_hidden,
                                                 forget_bias=forget_bias,
                                                 state_is_tuple=True,
                                                 cell_clip = cell_clip)
            
        
        self.embed_features_W = tf.get_variable("embed_features_W", 
                                    shape = [dim_image, dim_hidden],
                                    initializer = tf.contrib.layers.xavier_initializer())
        
        
        self.embed_features_b = tf.get_variable("embed_features_b", 
                                    shape = [dim_hidden],
                                    initializer = tf.contrib.layers.xavier_initializer())
            
        self.predict_word_W = tf.get_variable("predict_word_W", 
                                    shape = [dim_hidden, n_words],
                                    initializer = tf.contrib.layers.xavier_initializer())
        
        if bias_init_vector is not None:
            self.predict_word_b = tf.get_variable("predict_word_b",
                                    initializer = tf.constant(bias_init_vector,
                                                              dtype = tf.float32))
        else:
            self.predict_word_b = tf.get_variable("predict_word_b", 
                                    shape = [n_words],
                                    initializer = tf.contrib.layers.xavier_initializer())

    def build_model(self, stage):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_steps, self.dim_image])
        
        if stage == "training" or stage == "saliency":
            caption = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
            caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        #############feature embedding
        video_flat = tf.reshape(video, [-1, self.dim_image])
        features_emb = tf.nn.xw_plus_b(video_flat,
                                       self.embed_features_W,
                                       self.embed_features_b)
        features_emb = tf.reshape(features_emb,
                                  [self.batch_size, self.n_frame_steps, self.dim_hidden])
        #############feature embedding
        
        if stage == "training":
            self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell,
                                                           input_keep_prob = self.input_keep_prob,
                                                           output_keep_prob = self.output_keep_prob)

        state1 = self.lstm_cell.zero_state(self.batch_size, tf.float32)
        
        padding_frame = tf.zeros([self.batch_size, self.dim_hidden])
        padding_word = tf.zeros([self.batch_size, self.dim_word_emb])

        probs = []
        
        if stage == "inference":
            generated_words = []
        
        loss = 0.0
        
        
        saliency = []
        
        current_embed = tf.zeros([self.batch_size, self.dim_word_emb])
        
        for i in range(self.n_frame_steps):     #encode frame sequence
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("LSTM_CAP"):
                output1, state1 = self.lstm_cell(tf.concat([ features_emb[:,i,:], padding_word], 1),
                                                 state1)
            
        for i in range(self.n_lstm_steps):      #predict output sequence
            if i == 0:
                current_embed = tf.zeros([self.batch_size, self.dim_word_emb])
            with tf.variable_scope("LSTM_CAP", reuse=True):
                output1, state1 = self.lstm_cell(tf.concat([padding_frame, current_embed], 1),
                                                 state1)

            logit_words = tf.nn.xw_plus_b(output1, self.predict_word_W, self.predict_word_b)
            probs.append(logit_words)
            
            if stage == "training" or stage == "saliency":
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit_words,
                                                                               labels = caption[:,i])
                cross_entropy = cross_entropy * caption_mask[:,i]
                saliency.append(-cross_entropy)
                loss += tf.reduce_sum(cross_entropy)
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i])
            elif stage == "inference":          #greedy decoder
                max_prob_index = tf.argmax(logit_words, 1)
                generated_words.append(max_prob_index)
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                
        if stage == "training" or stage == "saliency":
            loss = loss / tf.reduce_sum(caption_mask)
            return loss, video, caption, caption_mask, saliency
        elif stage == "inference":
            return loss, video, generated_words, None, probs
        