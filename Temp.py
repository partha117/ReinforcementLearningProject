import torch
from torch import nn
from torch.nn.functional import softmax

def bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)
class AttLayer(nn.Module):
    def __init__(self, limit=768):
        super(AttLayer, self).__init__()
        self.limit = limit
        self.report_attention_layer = nn.Linear(in_features=self.limit, out_features=self.limit)
        self.report_linear_layer = nn.Linear(in_features=self.limit, out_features=self.limit)
        self.code_linear_layer = nn.Linear(in_features=self.limit, out_features=self.limit)


    def forward(self, x):
        report = x[:, 0, :self.limit]
        source = x[:, :, self.limit:]
        report_ll = self.report_linear_layer(report)
        code_ll = self.code_linear_layer(source)
        multiplication = torch.bmm(torch.unsqueeze(report_ll,1), code_ll.swapaxes(1,2))
        att_value = softmax(multiplication,dim=2)
        att_value = att_value.squeeze(1).unsqueeze(2)
        r_att_value = self.report_attention_layer(report)
        return att_value * source, r_att_value * report
class DQN(nn.Module):
    def __init__(self, limit=768):
        super(DQN, self).__init__()
        self.limit = limit
        self.attention = AttLayer(limit=self.limit)
        self.linear1 = nn.Linear(in_features=self.limit * 2, out_features=self.limit)
        self.linear2 = nn.Linear(in_features=self.limit, out_features=1)

    def forward(self, x):
        source, report = self.attention(x)
        x = torch.concat([source, torch.stack([report for i in range(source.shape[1])]).swapaxes(0,1)], axis=2)
        x = self.linear1(x)
        x = self.linear2(x)
        return softmax(x.squeeze(2), dim=1)
if __name__ == "__main__":
    a = torch.rand((5, 31,1,1536))
    a = torch.squeeze(a, 2)
    model = DQN()
    print(model(a))