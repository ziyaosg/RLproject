import os
import torch
import torch.nn as nn
from torch.nn.functional import softmax

from image_embed import ROOT_DIR, image_embed
from text_embed import TASK_DESCRIPTION, text_embed


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, queries, keys, values):
        queries = self.query_proj(queries)  # [batch_size, dim]
        keys = self.key_proj(keys)  # [batch_size, dim]
        values = self.value_proj(values)  # [batch_size, dim]

        attention_scores = torch.bmm(queries.unsqueeze(1), keys.unsqueeze(2))
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over keys
        attended_values = torch.bmm(attention_weights, values.unsqueeze(1)).squeeze(1)

        return attended_values


class MultimodalModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.cross_attention = CrossAttention(dim=embedding_dim)  # Set the embedding dimension

    def forward(self, text_embeddings, image_embeddings):
        combined_features = self.cross_attention(text_embeddings, image_embeddings, image_embeddings)
        return combined_features


def combine_embeddings(text_embedding, image_embedding):
    # combine embeddings
    model = MultimodalModel(embedding_dim=768)
    combined_embeddings = model(text_embedding, image_embedding[0])  # only using 1 pair for demo

    return combined_embeddings


if __name__ == "__main__":
    # generate text embedding
    text_embedding = text_embed(TASK_DESCRIPTION)

    # generate image embedding
    config = dict()
    config['model'] = 'vgg19'
    root_dir = ROOT_DIR
    dirs = os.listdir(root_dir)[:1]  # only using 1 for demo
    img_dirs = [os.path.join(root_dir, img_dir) for img_dir in dirs]

    img_paths = list()
    for img_dir in img_dirs:
        imgs_curr_dir = [os.path.join(img_dir, img_filename) for img_filename in os.listdir(img_dir)]
        img_paths.append(sorted(imgs_curr_dir))
    # print(img_paths)

    image_embeddings = list()
    for imgs_curr_dir in img_paths:
        image_embeddings.append(image_embed(imgs_curr_dir, config))

    # example usage
    embedding = combine_embeddings(text_embedding, image_embeddings[0])
    print("Combined Embeddings Shape:", embedding.shape)
    print(embedding)
