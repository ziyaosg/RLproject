import os
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim


from image_embed import ROOT_DIR, image_embed
from text_embed import TASK_DESCRIPTION, text_embed
from cross_attention import combine_embeddings


LABELS = x = torch.tensor([1])


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance between outputs
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        # Contrastive loss
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class SimpleEmbeddingNet(nn.Module):
    def __init__(self):
        super(SimpleEmbeddingNet, self).__init__()
        self.fc = nn.Linear(768, 128)  # Reduce dimensionality for demonstration

    def forward(self, x):
        output = self.fc(x)
        output = nn.functional.normalize(output, p=2, dim=1)  # Normalize the output
        return output


if __name__ == '__main__':
    # example usage
    # generate text embedding
    text_embeddings = text_embed(TASK_DESCRIPTION)

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
    embeddings = combine_embeddings(text_embeddings, image_embeddings[0])
    labels = LABELS

    # Create a dataset and dataloader for processing
    dataset = torch.utils.data.TensorDataset(embeddings, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model and loss function
    model = SimpleEmbeddingNet()
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(10):  # number of epochs
        for data in dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[0].unsqueeze(0), outputs[1].unsqueeze(0), labels[0])
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
