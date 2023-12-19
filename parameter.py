class parameters():
    def __init__(self,
        data_path =  'SLP/danaLab',                      
        batch_size = 30 ,                                                            
        cover_list =  ['uncover', 'cover1', 'cover2'],
        end_epoch =  100,
        epoch_step = -1,
        fc_depth =  50.0,
        use_bounding_box = False,    
        flip_test =  True,   
        pin_memory = True,                   
        perform_test = False,                                     
        num_channels = 1,             
        input_dimensions = (256, 256),               
        learning_rate = 0.001,
        lr_decay_epochs = [30, 40],         
        lr_decay_factor = 0.1,         
        source_modes = ['IR'], 
        model = 'StackedHourGlass',
        test_name = 'SLP-danaLab_exp',
        optimizer = 'adam',  
        output_shape = [64, 64, -1] ,
        output_dir = 'output/',
        logging_frequency = 10,   
        start_epoch = -1,              
        patch_size = (256, 256),
        trainIter = -1,
        use_gpu = False,
        nChannels = 256, 
        nStack = 2,
        nModules = 2,
        numReductions = 4, 
        robust = True,
        mu_loss = True):

        self.nChannels = nChannels
        self.numReductions = numReductions
        self.nStack = nStack
        self.nModules = nModules
        self.use_gpu = False
        self.data_path = data_path                
        self.batch_size = batch_size                                                                             
        self.cover_list =  cover_list
        self.end_epoch =  end_epoch                           
        self.epoch_step = epoch_step                            
        self.fc_depth =  fc_depth                                                    
        self.use_bounding_box = use_bounding_box                                                    
        self.flip_test =  flip_test                           
        self.pin_memory = pin_memory                          
        self.perform_test = perform_test                                             
        self.num_channels = num_channels                            
        self.input_dimensions = input_dimensions                    
        self.learning_rate = learning_rate                         
        self.lr_decay_epochs = lr_decay_epochs                     
        self.lr_decay_factor = lr_decay_factor                                             
        self.source_modes = source_modes                        
        self.model = model                                                           
        self.test_name = test_name                                                                 
        self.optimizer = optimizer                          
        self.output_shape = output_shape                                             
        self.logging_frequency = logging_frequency                                                       
        self.start_epoch = start_epoch                                                
        self.patch_size = patch_size                                                                                         
        self.trainIter = trainIter                                                                     
        self.use_gpu = use_gpu
        self.robust = robust
        self.mu_loss = mu_loss
        self.output_dir = output_dir

    def files(self):
      num_channels = 0
      for mod in self.source_modes:
        if 'RGB' in mod:
          num_channels+=3
        else:
          num_channels += 1
      self.num_channels = num_channels
      covStr = ''
      if 'uncover' in self.cover_list:
        covStr += 'u'
      if 'cover1' in self.cover_list:
        covStr += '1'
      if 'cover2' in self.cover_list:
        covStr += '2'
      
      if self.mu_loss:
        loss = 'multi_loss'
      else:
        loss = 'singel_loss'


      modalities = '_'.join(self.source_modes)
      self.exp_dir =  self.output_dir + self.model + '_' + str(self.nChannels) + '_' + str(self.nStack) + '_' + str(self.nModules) + '_' + str(self.numReductions) + loss +'/SLP' + '_' + modalities + '___' + covStr
      self.vis_dir = self.exp_dir + '/vis'
      self.vis_test_dir = self.vis_dir + '/test'
      self.vis_val_dir = self.vis_dir + '/val'  
      self.rst_dir = self.exp_dir + '/result'
      self.model_dir = self.exp_dir + '/model_dump'
      
      make_folder(self.model_dir)
      make_folder(self.vis_dir)
      make_folder(self.rst_dir)

import os
def make_folder(folder_name):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
