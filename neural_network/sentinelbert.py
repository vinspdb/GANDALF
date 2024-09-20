import torch.nn as nn


class BertBinaryClassificationHeads(nn.Module):
    def __init__(self, gpt_model, layer_sizes, dropout_probs):
        super(BertBinaryClassificationHeads, self).__init__()
        self.gpt_model = gpt_model
        assert len(layer_sizes) - 1 == len(dropout_probs), "Mismatch between layer sizes and dropout probabilities"
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_probs[i]))

        self.output_layer = nn.Linear(layer_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_ids, attention_mask):
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs[1]
        for layer in self.layers:
            x = layer(x)
        output_layer = self.output_layer(x)
        output_layer = self.sigmoid(output_layer)
        return output_layer.squeeze(1)