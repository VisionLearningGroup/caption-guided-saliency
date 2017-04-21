class flickr_cfg():
    def __init__(self):
        self.id = "Flickr30k"
        ############### Global Parameters ##############
        self.path_to_descriptors = "./DATA/Flickr30k/flickr30k_incp3/"
        self.path_to_images = "./DATA/Flickr30k/flickr30k-images/"
        self.descriptor_suffix = "_incp_v3"  
        
        ############### Split parameters ###############
        self.annotations_path = "./DATA/Flickr30k/results_20130124.token"
        #splits are taken from 
        self.train_file = "./DATA/Flickr30k/imgs_train.txt"
        self.val_file = "./DATA/Flickr30k/imgs_val.txt"
        self.test_file = "./DATA/Flickr30k/imgs_test.txt" 
         
        self.experiment = "./experiments/flickr30k/"                    # files will be overwritten in the case of multiple runs
        self.vocab_path = self.experiment                    
        self.model_path = self.experiment                     
        self.results_path = self.experiment                  
        
        ############## Training Parameters #############
        self.word_count_threshold = 1
        self.learning_rate = 5e-4
        self.n_epochs = 10
        self.batch_size = 100
        self.max_to_keep = 40
        self.save_every_n_epoch = 0.2       # fractional number allows to save the model multiple times during every epoch
                                            # makes sense since by epoch here we understand all sentences for the training data (5 for each image)
        self.use_hard_cache = False         # instead of relying on system file cache explicitly adds all features to dictionary
                                            # this option is RAM-consuming and not effective, reader will be rewritten
                                               
        ############## Model Parameters ################
        self.dim_image = 2048        # dimensionality of input features (2048 for InceptionV3 and ResNet)
        self.dim_hidden= 1300        # LSTM hidden state size
        self.dim_word_emb = 300      # input word dimensionality
        self.n_frame_step = 64       # dimensionality of the used convolutional layer (8x8)
        self.n_lstm_step = 25        # affects training time, can be reduced since training sentences are significantly shorter 
        self.forget_bias=1
        self.cell_clip = None
        self.input_keep_prob = 0.5 
        self.output_keep_prob = 0.5
        
class msr_vtt_cfg():
    def __init__(self):
        self.id = "MSR-VTT"
        ############### Global Parameters ##############
        data_path = "./DATA/MSR-VTT/" 
        self.descriptor_suffix = "_incp_v3"
        
        ############### Split parameters ###############
        self.trainval_annotations = data_path + "train_val_videodatainfo.json"
        self.test_annotations = data_path + "test_videodatainfo.json"
        self.path_to_trainval_descriptors = data_path + "trainval_descriptors/"
        self.path_to_test_descriptors = data_path + "test_descriptors/"
        self.path_to_trainval_video = data_path + "TrainValVideo/"
        self.path_to_test_video = data_path + "TestVideo/"
        
        
        self.experiment = "./experiments/msr-vtt/"                      # files will be overwritten in the case of multiple runs
        self.vocab_path = self.experiment                     
        self.model_path = self.experiment                     
        self.results_path = self.experiment
        
        ############## Training Parameters #############
        self.word_count_threshold = 1
        self.learning_rate = 5e-4
        self.n_epochs = 10
        self.batch_size = 100
        self.max_to_keep = 30               # number of snapshots to keep
        self.save_every_n_epoch = 0.1       # fractional number allows to save the model multiple times during every epoch
                                            # makes sense since by epoch here we understand all sentences for the training data (20 for each video)
        self.use_hard_cache = True          # instead of relying on system file cache explicitly adds all features to dictionary
                                            # this option is RAM-consuming and not effective, reader will be rewritten   
        
        ############## Model Parameters ################
        self.dim_image = 2048        # dimensionality of input features (2048 for InceptionV3 and ResNet)
        self.dim_hidden= 1300        # LSTM hidden state size
        self.dim_word_emb = 300      # input word dimensionality
        self.n_frame_step = 26       # number of frames used for training
        self.n_lstm_step = 25        # affects training time, can be reduced since training sentences are significantly shorter 
        self.forget_bias=1
        self.cell_clip = None
        self.input_keep_prob = 0.5 
        self.output_keep_prob = 0.5