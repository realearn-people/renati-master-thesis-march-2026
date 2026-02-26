base_architecture = 'vgg16_bn' 
img_size = 224
prototype_shape = (60, 512, 1, 1)
num_classes = 6
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = 'Rav_frame_vgg16_bn 60 16'

data_path = './RAVDESS_dataset_v1/'
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
val_dir   = data_path + 'val/'          # <— 新增
train_push_dir = data_path + 'train/'
train_batch_size = 16
test_batch_size = 16
val_batch_size   = 16                   # <— 新增
train_push_batch_size = 16

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 100
num_warm_epochs = 5

push_start = 30
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
