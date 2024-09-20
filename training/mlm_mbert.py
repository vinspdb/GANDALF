import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import random
import torch
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

def pre_train(model, optimizer, train_dataloader, val_dataloader, epochs, patience):
        # Training loop
        patience_counter = 0
        best_val_loss = float('inf')
        for epoch in range(epochs):  # number of epochs
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask,
                                labels=labels)
                train_loss = outputs.loss
                accelerator.backward(train_loss)
                optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                cumulated_loss = torch.as_tensor([0.0]).to(accelerator.device)

                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    labels=labels)
                    loss = outputs.loss
                    #val_loss += loss.item()
                    cumulated_loss = cumulated_loss + loss

            cumulated_loss = accelerator.gather(cumulated_loss)
            if accelerator.is_main_process:
                cumulated_loss = cumulated_loss.cpu().mean().item()
                print(f"Epoch {epoch + 1} Validation Loss: {cumulated_loss/len(val_dataloader):.4f}")
                if cumulated_loss/len(val_dataloader) < best_val_loss:
                    accelerator.set_trigger()
                    best_val_loss = cumulated_loss/len(val_dataloader)
                    patience_counter = 0
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained('./mlm_base')
                    tokenizer.save_pretrained('./mlm_base')
                else:
                    patience_counter += 1
                scheduler.step(cumulated_loss / len(val_dataloader))
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            accelerator.wait_for_everyone()

def prepare_text(inputs):
        inputs['labels'] = inputs.input_ids.detach().clone()
        print(inputs.keys())

        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(inputs.input_ids.shape)
        # create mask array
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
                   (inputs.input_ids != 102) * (inputs.input_ids != 0)
        selection = []

        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103

        return inputs

class SentinelDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

if __name__ == '__main__':
    seed = 42
    LEARNING_RATE = 1e-5
    BATCH = 512
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

    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_train, y_train,
                                                                      test_size=0.2, random_state=seed,
                                                                      stratify=y_train)

    weights_dir = get_weight_dir('prajjwal1/bert-medium')

    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    model = AutoModelForMaskedLM.from_pretrained(weights_dir)

    inputs_train = tokenizer(X_train_new, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs_val = tokenizer(X_val_new, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    inputs_train = prepare_text(inputs_train)
    inputs_val = prepare_text(inputs_val)

    dataset_train = SentinelDataset(inputs_train)
    dataset_val = SentinelDataset(inputs_val)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH, shuffle=True)

    print('TRAINING START...')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(split_batches=True, kwargs_handlers=[ddp_kwargs])

    device = accelerator.device
    print('device-->', device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) #AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)

    # Early stopping parameters
    pre_train(model, optimizer, train_dataloader, val_dataloader, 30, 5)



