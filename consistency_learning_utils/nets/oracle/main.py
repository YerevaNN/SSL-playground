from oracle import get_dataset
from convnet import Oracle

if __name__ == "__main__":
    #csv_path = '/home/hkhachatrian/SSL-playground/session_data/5L53Vy04iImX6naGoqdy/base/widerperson_base_7_from_coco_0_feature_extraction_s7/predictions_on_unlabeled.csv'
    csv_path = '/home/hkhachatrian/SSL-playground/session_data/5L53Vy04iImX6naGoqdy/base/widerperson_base_3_from_coco_0_phd_fe_teacher_boxes/predictions_on_unlabeled.csv'
    label_root = '/home/hkhachatrian/SSL-playground/session_data/5L53Vy04iImX6naGoqdy/base/labels/'

    samples, selected_pseudo_labels, train_cl_masks, test_cl_masks = get_dataset(csv_path, label_root, 3700)

    model = Oracle('wp_stage3_phd_teacher_boxes')
   # model.load_from_path('/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/checkpoints_nightowls/last_200.ckpt')
    model.set_datasets(samples, selected_pseudo_labels, train_cl_masks, test_cl_masks, 3700)
    model.fit_model()
    model.test()
