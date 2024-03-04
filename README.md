# Source code overview
### configs: store configurations for training of corresponding backbones and inference.

### logs: place for training and testing logs.

### notebooks: place for jupyter notebooks that are used to train on hosted platforms such as Kaggle or Colab. Code in these notebooks is merged from src.

### src: contains 4 different package:
  + data: includes modules used for data preprocessing, dataset generating, etc.
  + modelling: includes backbone and derived arch from that backbone. (VGG -> VGG11, VGG13, VGG16, ...)
  + tools: main operations (e.g. train, inference, visualize, etc.)
  + utils: processing functions that are called by modules from data, tools, or main.py

# Training/ Testing configurations
  + EXPERIMENTS: training's metadata.
  + DATA:
    + DATASET_NAME: name of dataset used to train on.
    + INPUT_SHAPE: backbone's input shape (e.g. VGG: 3x224x224, YOLOv8: 3x640x640).
    + TRAIN_SIZE: train vs val splitting ratio. 
    + BATCH_SIZE: should be set to max GPU utilization.
    + NUM_WORKERS: # of parallel processes will be spawned. (Note: spawned process will take up RAM the same as main process. Use it at your own risk :\)\) ).
    + TRANSFORM (optional): Data transformation applies to X or variables, including NAME_LIST for transformation name and ARGS for corresponding parameters. Reference usage at https://pytorch.org/vision/stable/transforms.html. If you don't use any transformation, just don't specify it in your config.
    + TARGET_TRANSFORM (optional): The same as above but used for label.
  + CHECKPOINT:
    + SAVE: checkpointing or not.
    + LOAD: load your trained model in checkpoint directory.
    + SAVE_ALL: brute-force saving model method.
    + RESUME_NAME: Checkpoint name that's gonna continue to train.
  + EPOCH:
    + START: starting epoch.
    + EPOCHS: how many epoch you're gonna train in this session.
  + METRICS: NAME_LIST and its related ARGS. Reference at https://pytorch.org/torcheval/stable/
  + SOLVER: indispensable and intimately related components when training or model
    + MODEL:
        + BASE: backbone name
        + NAME: derived arch from predefined backbone
        + PRETRAINED: use pretrained weight from Torch hub
        + ARGS: kwargs passed into backbone<br>
    
      Reference: https://pytorch.org/vision/stable/models.html

    
    + OPTIMIZER:
        + NAME: optimizer name
        + ARGS: kwargs for chosen optimizer  
        Reference: https://pytorch.org/vision/stable/models.html (Vision model)
    + LR_SCHEDULER:
        + NAME: Lr scheduler name
        + ARGS: kwargs for chosen lr scheduler
    + LOSS:
        + NAME: Loss function name
        + ARGS: kwargs for chosen loss function
    + EARL STOPPING: based on the difference between train loss & val loss.
        + PATIENCE: How many epoch for training process to continue when val loss > train loss
        + MIN_DELTA: additional tolerance gap between train and val loss. 
    + MISC: comprises miscellaneous configs

    **Note**: In case of using custom version of any part in SOLVER, you must provide it in MODELLING dir.

### How to train model with this source
    Pipeline: Prep dataset -> Set configs -> Train model