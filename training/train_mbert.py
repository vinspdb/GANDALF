import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from neural_network.historydataset import TextDataset
from neural_network.sentinelbert import BertBinaryClassificationHeads
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from sklearn.metrics import confusion_matrix
from accelerate import DistributedDataParallelKwargs, Accelerator
import os
from pathlib import Path

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)

HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")


def get_weight_dir(
        model_ref: str,
        *,
        model_dir=HF_DEFAULT_HOME,
        revision: str = "main", ) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir()
    model_path = model_dir / "--".join(["models", *model_ref.split("/")])
    assert model_path.is_dir()
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir()
    return weight_dir


def fine_tuning(model, optimizer, train_dataloader, val_dataloader, scheduler, patience, epochs, best_f_score, model_name):
        # Training loop
        patience_counter = 0
        for epoch in range(epochs):  # number of epochs
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].type(torch.FloatTensor)
                labels = labels.to(device)
                output = model(input_ids, attention_mask)
                loss = criterion(output, labels)
                accelerator.backward(loss)
                optimizer.step()
            # Validation phase
            model.eval()
            all_predictions = []
            all_targets = []
            with torch.no_grad():
                cumulated_loss = torch.as_tensor([0.0]).to(accelerator.device)
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].type(torch.FloatTensor)
                    labels = labels.to(device)
                    output = model(input_ids, attention_mask)
                    loss = criterion(output, labels)
                    cumulated_loss = cumulated_loss + loss
                    predicted = (output > 0.5).float()
                    all_targets.extend(batch['labels'].to('cpu').numpy())
                    all_predictions.extend(predicted.to('cpu').numpy())
            tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
            tn = torch.as_tensor([tn]).to(accelerator.device)
            fp = torch.as_tensor([fp]).to(accelerator.device)
            fn = torch.as_tensor([fn]).to(accelerator.device)
            tp = torch.as_tensor([tp]).to(accelerator.device)

            cumulated_tn = accelerator.gather(tn)
            cumulated_fp = accelerator.gather(fp)
            cumulated_fn = accelerator.gather(fn)
            cumulated_tp = accelerator.gather(tp)
            cumulated_loss = accelerator.gather(cumulated_loss)

            if accelerator.is_main_process:
                cumulated_tn = cumulated_tn.cpu().sum().item()
                cumulated_fp = cumulated_fp.cpu().sum().item()
                cumulated_fn = cumulated_fn.cpu().sum().item()
                cumulated_tp = cumulated_tp.cpu().sum().item()
                cumulated_loss = cumulated_loss.cpu().mean().item()

                precision = 0 if cumulated_tp + cumulated_fp == 0 else cumulated_tp / (cumulated_tp + cumulated_fp)
                recall = 0 if cumulated_tp + cumulated_fp == 0 else cumulated_tp / (cumulated_tp + cumulated_fn)
                f1 = 0 if precision + recall == 0 else 2 * ((precision * recall) / (precision + recall))
                print(f"Epoch {epoch + 1}, Validation Loss: {cumulated_loss/len(val_dataloader):.4f}, F1 score val: {f1:.4f}")
                if f1 > best_f_score:
                    accelerator.set_trigger()
                    best_f_score = f1
                    patience_counter = 0
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(unwrapped_model.state_dict(), model_name + str(epoch + 1) + '.pth')
                else:
                    patience_counter += 1
                scheduler.step(cumulated_loss / len(val_dataloader))
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            accelerator.wait_for_everyone()


# Function to count the number of layers
def count_layers(model):
    return len(list(model.modules())) - 1  # Subtract 1 to exclude the top-level module itself



if __name__ == '__main__':
    seed = 42
    LEARNING_RATE = 1e-5  # 0.00001

    BATCH = 1024
    dataset = 'CZ_200_Settembre_Tiling-2'
    set_seed(seed)
    with open(dataset+'/train/train' + '.pkl', 'rb') as f:
        train = pickle.load(f)

    mask_train = pd.read_csv(dataset+'/train/masks_12.csv', header=None)

    
    X_train, X_val, y_train, y_val = train_test_split(train,
                                                      mask_train.to_numpy().reshape(len(mask_train.to_numpy()), ),
                                                      test_size=0.5, random_state=seed,
                                                      stratify=mask_train.to_numpy().reshape(
                                                          len(mask_train.to_numpy()), ))
    

    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_val,y_val, test_size=0.2, random_state=seed, stratify=y_val)

    weights_dir = get_weight_dir('prajjwal1/bert-medium')

    tokenizer = AutoTokenizer.from_pretrained(weights_dir)

    model = AutoModel.from_pretrained(weights_dir)

    train_dataset = TextDataset(X_train_new, y_train_new, tokenizer, 512)
    val_dataset = TextDataset(X_val_new, y_val_new, tokenizer, 512)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    print('TRAINING START...')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(split_batches=True, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    print('device-->', device)
    layer_sizes = [model.config.hidden_size, 128]  # Example sizes for each layer
    dropout_probs = [0.2]  # Dropout probabilities for each layer
    model = BertBinaryClassificationHeads(model, layer_sizes, dropout_probs).to(device)
    layers_to_freeze = 0
    criterion =  nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    startTime = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    fine_tuning(model, optimizer, train_dataloader, val_dataloader, scheduler, 1, 2, -float('inf'), dataset+'_'+str(layers_to_freeze)+'layers_orig_')


