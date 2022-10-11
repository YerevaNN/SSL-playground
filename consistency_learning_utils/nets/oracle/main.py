from convnet import Oracle

if __name__ == "__main__":
    label_root = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels/'
    image_paths = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels/'
    # feature_data_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/teacher_boxes_nightowls_unlabeled_vs_gt.txt'
    feature_data_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/vis_drone_base_3_from_coco_0_convnet_on_vis_drone_3_newSynthetics.txt'
    checkpoint_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/checkpoints/fair_coco_0/our_best.ckpt'
    skip_data_path = '/lwll/external/nightowls/validation.txt'

    class_num = 10
    box_score_thresh = 0.005
    lr = 0.01
    experiment_name = 'new_convnet_vis_drone'
    num_epochs = 50
    # experiment_name = 'debug'

    model = Oracle(experiment_name, lr = lr, class_num=class_num, train_epochs=num_epochs)
    # model.load_from_path(checkpoint_path)

    split = 0.95
    batch_size = 16
    model.set_datasets(feature_data_path, label_root, split, batch_size, class_num, box_score_thresh, skip_data_path)
    # model.test()
    model.fit_model()

