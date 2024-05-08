import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


TASK_DESCRIPTION = ("In the workspace, an operation bench is equipped with four distinctly colored holes: "
                    "red, blue, green, and yellow. "
                    "Each hole is designed to accommodate a ball of its corresponding color. "
                    "Initially, the placement decision for each colored ball into its respective hole is set randomly. "
                    "The robot arm is tasked with the correct placement of each colored ball into the matching colored hole. "
                    "However, the robot does not have pre-programmed instructions for this initial setup. "
                    "Instead, it relies on gesture instructions provided by a human operator to determine the correct placement sequence. "
                    "The goal for the robot arm is to accurately interpret these human gestures to identify and execute the appropriate action for placing each colored ball into the correct colored hole, "
                    "thus ensuring successful task completion.")


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    pooled_outputs = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    return pooled_outputs


def text_embed(task_description):
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
        model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    except Exception as e:
        print("Failed to load model or tokenizer:", str(e))
        exit(1)

    # Tokenization and model input preparation
    max_length = 512  # Adjusted max_length for a single input
    input_text = [task_description]  # Single input text
    batch_dict = tokenizer(input_text, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

    if torch.cuda.is_available():
        model = model.cuda()  # Move model to GPU
        batch_dict = {k: v.cuda() for k, v in batch_dict.items()}  # Move inputs to GPU
    else:
        print("CUDA is not available, running on CPU.")

    # Forward pass through the model
    outputs = model(**batch_dict)
    embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # Normalize embedding
    embedding = F.normalize(embedding, p=2, dim=1)

    # Output the embedding for further processing
    # print("Task Description Embedding:", embedding)
    return embedding


if __name__ == '__main__':
    print(text_embed(TASK_DESCRIPTION))
