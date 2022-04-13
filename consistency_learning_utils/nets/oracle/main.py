from convnet import Oracle

if __name__ == "__main__":
    label_root = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels/'
    image_paths = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels/'
    feature_data_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/night_teacher_boxes_vs_gt.txt'
    checkpoint_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/checkpoints/coco_augmented_weak_nightowls/best.ckpt'
    
    class_num = 3
    box_score_thresh = 0.05
    lr = 0.001
    experiment_name = 'classificator_on_night_0'
    # experiment_name = 'debug'

    model = Oracle(experiment_name, lr = lr, class_num=class_num)
    # model.load_from_path(checkpoint_path)

    split = 0.99
    batch_size = 16
    model.set_datasets(feature_data_path, label_root, split, batch_size, class_num, box_score_thresh)
    # model.test()
    model.fit_model()

