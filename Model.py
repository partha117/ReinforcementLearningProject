from torch import nn
import torch
import transformers

def get_last_layer_output(tensor, num_sentences, max_segments):
    idxs_sentences = torch.arange(num_sentences)
    idx_last_output = max_segments - 1
    return tensor[idxs_sentences, idx_last_output]


class ClassificationHead(nn.Module):
    def __init__(self, input_size, embed_size, num_labels=2):
        super().__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.embed_size = embed_size
        self.code_lstm = nn.LSTM(self.input_size, self.embed_size, 1, bias=True, batch_first=True)
        self.report_lstm = nn.LSTM(self.input_size, self.embed_size, 1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(0.15)
        self.out_proj = nn.Linear(self.embed_size * 2, self.num_labels)

    def forward(self, code_hidden_states, report_hidden_states, code_properties, report_properties):
        code_lstm_output, _ = self.code_lstm(code_hidden_states)
        report_lstm_output, _ = self.report_lstm(report_hidden_states)
        code_last_layer = get_last_layer_output(self.dropout(code_lstm_output), code_properties['num_sentences'],
                                                code_properties['max_segments'])
        report_last_layer = get_last_layer_output(self.dropout(report_lstm_output), report_properties['num_sentences'],
                                                  report_properties['max_segments'])
        combined_input = torch.cat([code_last_layer, report_last_layer], dim=1)
        logits = self.out_proj(combined_input)
        return logits


class ClassifierModel(nn.Module):

    def __init__(self, config, num_labels, base_model, embed_size,
                 code_overlap_size=0, report_overlap_size=0):
        super(ClassifierModel, self).__init__()
        self.num_labels = num_labels
        self.config = config
        self.transformer = base_model
        self.code_max_size = 2048 if isinstance(base_model, transformers.ReformerModel) else config.hidden_size
        self.report_max_size = 2048 if isinstance(base_model, transformers.ReformerModel) else config.hidden_size
        self.code_overlap_size = code_overlap_size
        self.report_overlap_size = report_overlap_size
        self.dropout = nn.Dropout(0.2)
        self.classifier = ClassificationHead(
            input_size=self.code_max_size,
            embed_size=embed_size, num_labels=num_labels)

    def forward(self, code_input_id, code_attention_mask, report_input_id, report_attention_mask, code_properties,
                report_properties):
        # c_in = get_splitted_tensor(code_input_id, max_size=self.code_max_size, overlap_size=self.code_overlap_size)
        # r_in = get_splitted_tensor(report_input_id, max_size=self.report_max_size,
        #                            overlap_size=self.report_overlap_size)
        #
        # c_in_, c_num_sentences, c_max_segments = reshape_input(c_in)
        # r_in_, r_num_sentences, r_max_segments = reshape_input(r_in)

        code_output = self.dropout(self.transformer(input_ids=code_input_id, attention_mask=code_attention_mask)[1])
        report_output = self.dropout(
            self.transformer(input_ids=report_input_id, attention_mask=report_attention_mask)[1])

        converted_c_out = code_output.view(code_properties['num_sentences'], code_properties['max_segments'],
                                           code_output.shape[-1])
        converted_r_out = report_output.view(report_properties['num_sentences'], report_properties['max_segments'],
                                             report_output.shape[-1])
        output = self.classifier(converted_c_out, converted_r_out,
                                 code_properties=code_properties,
                                 report_properties=report_properties)
        return output
