from oracle import get_dataset
from convnet import Oracle

if __name__ == "__main__":
    # zarr_path = '/home/hkhachatrian/SSL-playground/widerperson_base_3_from_coco_0_zarr_extended_features_0/features/features.zarr'
    # label_root = '/home/hkhachatrian/SSL-playground/session_data/5L53Vy04iImX6naGoqdy/base/labels/'
    zarr_path = '/home/hkhachatrian/SSL-playground/nightowls_adaption_3_from_coco_0_zarr_extended_features_nightowls/features/features.zarr'
    label_root = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels/'

    samples = get_dataset(zarr_path)

    lr = 0.0001
    model = Oracle('zarr_extended_features_night_save_best', zarr_path = zarr_path, lr = lr)
    split = 0.9
    batch_size = 32
    model.set_datasets(samples, label_root, split, batch_size)
    model.fit_model()