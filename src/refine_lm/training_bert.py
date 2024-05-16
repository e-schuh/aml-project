
import argparse
import os
import torch


#from transformers import *
from redubias.calc_bias import Dataset, calculate_reward, collate_fn
from model_BERT import CustomBERTModel
import _pickle as pickle
from time import process_time
from templates.lists import Lists
from transformers import set_seed

gpuid = -1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_LEVEL_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', type=int, default=0,
                    help='select GPU')
    parser.add_argument('--mini_batch_size', type=int, default=70, help="Size of batch per update")
    parser.add_argument('--batch_size', type=int, default=70, help = "Samples per Template")
    parser.add_argument('--lr', type=float, default=5e-9, help = "Learning Rate")
    parser.add_argument('--topk', type=int, default=8, help = "TopK")
    parser.add_argument('--model_name', type=str, default="new_model", help = "Name of the model")
    parser.add_argument('--ppdata', type=str, default="", help = "Path of the preprocessed data")
    parser.add_argument('--use_he_she', help="Whether to account for lm predictions on he/she", type=int, default=1)
    parser.add_argument('--intrasentence_model', help="Which intrasentence model to use", type=str, default = "SwissBertForMLM")
    parser.add_argument('--pretrained_model_name', help="Which intrasentence model to use", type=str, default = "ZurichNLP/swissbert-xlm-vocab")
    args = parser.parse_args()
    # torch.cuda.set_device(args.device) # Googling it seems this doesn't work on CPU-only devices
    ppdata_path = args.ppdata
    ## Loading Pre-processed Data

    lists = Lists("word_lists", None)


    set_seed(0)


    with open(ppdata_path, 'rb') as file:
        pp_data = pickle.load(file)

    keys = list(pp_data.keys())
    values = list(pp_data.values())
    values = [pp_data[k] for k in keys]

    batch_size = args.batch_size
    training_values = Dataset(values)
    training_generator = torch.utils.data.DataLoader(training_values, batch_size=batch_size, collate_fn=collate_fn, num_workers=2)
    
    #######VARIABLES########
    mini_batch_size = args.mini_batch_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr
    topk = args.topk
    model_name = args.model_name
    """Defining Model"""
    print("Defining Model")
    #NOTE: Insert the name of the model here
    model = CustomBERTModel(topk, batch_size, args.intrasentence_model, args.pretrained_model_name).to(DEVICE)

    for layer_name, param in model.named_parameters():
        if 'bert' in layer_name:   # Still works with our custom model
            param.requires_grad = False
    print("Loading Dataset")
    training_values = Dataset(values)

    learning_rate = lr
    print("Number of Samples: ", len(training_generator))
    print(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(training_generator)
    print("Number of training steps : ",num_training_steps)
    step = 0
    model.train()
    rewards = []
    print("Starting training")
    start = process_time()
    cp_path = 'data/logs/training_logs/'+model_name + '.pt'
    print('Name:', model_name)
    for epoch in range(num_epochs):
        # Training
        for local_batch in training_generator:
            with torch.device(DEVICE): # Commented out cause probably doesn't work on CPU-only device
            # Transfer to GPU
                loss, reward = calculate_reward(args, local_batch, mini_batch_size, topk, model.tokenizer, model)
                if step % 100 == 0:
                    print("Step ", step, "| Loss:  ", loss.item(), "| Reward: ", reward, flush=True)
                optimizer.zero_grad()
                if not torch.isnan(loss):
                    loss.backward()
                optimizer.step()
                rewards.append(reward)

                step += 1

                # Break out of training loop for development purposes
                if step % 1==0:
                    break

    stop = process_time()
    print("time elapsed: ", stop - start)
    #Save model
    save_dir = os.path.join(TOP_LEVEL_DIR, "data/refine_lm/saved_models/tiny")
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{os.path.join(save_dir, model_name)}.pth'
    cfg = {
    'state_dict': model.state_dict(),
    'topk': model.topk,
    'batch_size': model.batch_size,
    }
    torch.save(cfg, save_path)

if __name__ == '__main__':
    # mp.set_start_method('spawn')  # seems to be needed otherwise we get error with os.fork and JAX incompatibility. 
    main()
