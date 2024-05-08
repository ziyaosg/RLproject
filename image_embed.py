import os
import cv2
import torch
import numpy as np
import utils.utils as utils


ROOT_DIR = '/Users/ziyaoshangguan/Documents/jester/20bn-jester-v1'


def max_pool_images(image_paths):
    images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]
    if not images or any(img.shape != images[0].shape for img in images):
        raise ValueError("All images must have the same dimensions and number of channels.")
    stacked_images = np.stack(images, axis=0)
    max_pooled_image = np.max(stacked_images, axis=0)
    result_image = cv2.cvtColor(max_pooled_image, cv2.COLOR_RGB2BGR)

    return result_image


def image_embed(img_paths, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names \
        = utils.prepare_model(config['model'], device)

    img = max_pool_images(img_paths)
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range

    content_img = utils.prepare_img(img, device)
    content_img_set_of_feature_maps = neural_net(content_img)
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    embedding = target_content_representation.view(-1)
    return embedding


def main(config):
    root_dir = ROOT_DIR
    dirs = os.listdir(root_dir)[:1]     # only using 1 for demo
    img_dirs = [os.path.join(root_dir, img_dir) for img_dir in dirs]

    img_paths = list()
    for img_dir in img_dirs:
        imgs_curr_dir = [os.path.join(img_dir, img_filename) for img_filename in os.listdir(img_dir)]
        img_paths.append(sorted(imgs_curr_dir))

    # print(img_paths)
    embeddings = list()
    for imgs_curr_dir in img_paths:
        embeddings.append(image_embed(imgs_curr_dir, config))
    return embeddings


if __name__ == '__main__':
    config = dict()
    config['model'] = 'vgg19'
    print(main(config))
