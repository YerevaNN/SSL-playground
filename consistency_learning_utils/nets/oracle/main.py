from convnet import Oracle

if __name__ == "__main__":
    # zarr_path = '/home/hkhachatrian/SSL-playground/widerperson_base_3_from_coco_0_zarr_extended_features_0/features/features.zarr'
    label_root = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels/'
    # zarr_path = '/home/hkhachatrian/SSL-playground/nightowls_adaption_3_from_coco_0_zarr_extended_features_nightowls/features/features.zarr'
    image_paths = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels/'
    feature_data_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/nightowls_gt_augmented_stronger.txt'
    checkpoint_path = '/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/checkpoints/coco_augmented/best.ckpt'
    class_num = 3
    box_score_thresh = 0.005

    lr = 0.001
    model = Oracle('coco_augmented_strong_nightowls', lr = lr)
    model.load_from_path(checkpoint_path)
    split = 0.95
    batch_size = 16
    model.set_datasets(feature_data_path, label_root, split, batch_size, class_num, box_score_thresh)
    model.fit_model()
