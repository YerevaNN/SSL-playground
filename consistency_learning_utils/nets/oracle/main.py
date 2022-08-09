from convnet import Oracle

if __name__ == "__main__":
    label_root = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels/'
    image_paths = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels/'
    # feature_data_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/teacher_boxes_nightowls_unlabeled_vs_gt.txt'
    feature_data_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/teacher_boxes_nightowls_labeled_vs_gt_with_validation.txt'
    checkpoint_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/checkpoints/fair_coco_0/our_best.ckpt'
    skip_data_path = '/lwll/external/nightowls/validation.txt'

    class_num = 3
    box_score_thresh = 0.005
    lr = 0.001
    experiment_name = 'fair_night_pretrained'
    # experiment_name = 'debug'

    model = Oracle(experiment_name, lr = lr, class_num=class_num)
    model.load_from_path(checkpoint_path)

    split = 0.1
    batch_size = 16
    model.set_datasets(feature_data_path, label_root, split, batch_size, class_num, box_score_thresh, skip_data_path)
    # model.test()
    model.fit_model()

