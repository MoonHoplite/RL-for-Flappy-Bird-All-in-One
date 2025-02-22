import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Dueling(nn.Module):
    def __init__(self, n_actions):
        super(Dueling, self).__init__()
        self.fc_A = torch.nn.Linear(512, n_actions)
        self.fc_V = torch.nn.Linear(512, 1)

    def forward(self, x):
        a = self.fc_A(x)
        v = self.fc_V(x)
        q = v + a - a.mean(1).view(-1, 1)
        return q


class DQN_q(nn.Module):
    def __init__(self, input_shape: tuple=(4, 80, 80), output_shape: int=2, dueling: bool=False) -> None:
        super(DQN_q, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_shape = self._get_conv_output(input_shape)
        if dueling:
            self.fc = nn.Sequential(
                nn.Linear(conv_out_shape, 512),
                nn.ReLU(),
                Dueling(output_shape)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(conv_out_shape, 512),
                nn.ReLU(),
                nn.Linear(512, output_shape)
            )
        
    def _get_conv_output(self, shape: tuple) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(o.view(1, -1).size(1))
        
    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)
    
    
class VPG_p(nn.Module):
    def __init__(self, input_shape: tuple=(4, 80, 80), output_shape: int=2) -> None:
        super(VPG_p, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_shape = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.last = nn.Linear(512, output_shape)
        self.last.bias.data.fill_(0)
        self.last.bias.data[0] = 0.0  # 不跳跃的初始偏置
        self.last.bias.data[1] = -2.   # 跳跃的初始偏置
        
        
    def _get_conv_output(self, shape: tuple) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(o.view(1, -1).size(1))
        
    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        logits = self.last(self.fc(conv_out))
        
        return F.softmax(logits, dim=-1)