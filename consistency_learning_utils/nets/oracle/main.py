from oracle import get_dataset
from convnet import Oracle

if __name__ == "__main__":
    csv_folder_path = '/home/hkhachatrian/SSL-playground/session_data/5L53Vy04iImX6naGoqdy/base/widerperson_base_3_from_coco_0_extended_features_separate_0/features/'
    label_root = '/home/hkhachatrian/SSL-playground/session_data/5L53Vy04iImX6naGoqdy/base/labels/'

    samples = get_dataset(csv_folder_path)

    model = Oracle('extended_features_0_bs4', csv_folder_path = csv_folder_path)
    split = 0.9
    batch_size = 4
    model.set_datasets(samples, label_root, split, batch_size)
    model.fit_model()