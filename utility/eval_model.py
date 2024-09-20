from transformers import AutoModel, AutoTokenizer
from neural_network.historydataset import TextDataset
from neural_network.sentinelbert import BertBinaryClassificationHeads
from torch.utils.data import DataLoader
import torch
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = 'CZ_200_Settembre_Tiling-2'

import os
from pathlib import Path
from typing import Optional

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


weights_dir = get_weight_dir('prajjwal1/bert-medium')

with open(dataset+'/test/test' + '.pkl', 'rb') as f:
    X_test = pickle.load(f)

mask_test = pd.read_csv(dataset+'/test/masks_12.csv', header=None)
tokenizer = AutoTokenizer.from_pretrained(weights_dir)

test_dataset = TextDataset(X_test, mask_test.to_numpy().reshape(len(mask_test.to_numpy()),), tokenizer, 512)

test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
test_model = AutoModel.from_pretrained(weights_dir)

layer_sizes = [test_model.config.hidden_size, 128]  # Example sizes for each layer
dropout_probs = [0.2] # Dropout probabilities for each layer
test_model = BertBinaryClassificationHeads(test_model, layer_sizes, dropout_probs).to(device)
test_model.load_state_dict(torch.load(dataset+'_0layers_orig_2.pth'))
test_model = test_model.to(device)

test_model.eval()

all_targets = []
all_predictions = []
pred_prob = []
print('TESTING')

with torch.no_grad():  # non aggiorna i pesi
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = test_model(input_ids, attention_mask)
        predicted = (output > 0.5).float()
        all_targets.extend(batch['labels'].to('cpu').numpy())
        all_predictions.extend(predicted.to('cpu').numpy())

all_targets = [int(x) for x in all_targets]
all_predictions = [int(x) for x in all_predictions]

f1_1 = classification_report(all_targets, all_predictions, output_dict=False, digits=4)
print(f1_1)

tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions,).ravel()

result = open(dataset + '_report.txt', 'w')
result.write(f1_1)
result.write('\n'+str(tn)+' '+str(fp)+' '+str(fn)+' '+str(tp))
