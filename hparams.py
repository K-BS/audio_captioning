from pathlib import Path

class hparams:
    batch_size = 16
    nhead = 12
    nhid = 768
    nlayers = 6
    ninp = 64
    ntoken = 4370 + 1
    clip_grad = 2.5
    lr = 5e-4  # learning rate
    #5e-5#3e-4
    beam_width = 3
    training_epochs = 300
    log_interval = 100
    checkpoint_save_interval = 5

    seed = 1111
    device = 'cuda'   #'cuda:0' 'cuda:1' 'cpu'
    mode = 'eval'

    name = '2_28'

    nkeyword = 4979

    label_smoothing = True
    load_pretrain_cnn = True
    load_pretrain_emb = True
    load_pretrain_model = False
    spec_augmentation = True
    scheduler_decay = 0.98

    # data(default)
    data_dir = Path(r'./create_dataset/data/data_splits')
    eval_data_dir = r'./create_dataset/data/data_splits/evaluation'
    train_data_dir = r'./create_dataset/data/data_splits/development'
    test_data_dir = r'./create_dataset/data/test_data'
    word_dict_pickle_path = r'./create_dataset/data/pickles/words_list.p'
    word_freq_pickle_path = r'./create_dataset/data/pickles/words_frequencies.p'


    # pretrain_model
    pretrain_emb_path = r'w2v_192.mod'
    pretrain_cnn_path = r'audioset_10_10_0.4593.pth'
    pretrain_model_path = r'models/1_2/10.pt'
