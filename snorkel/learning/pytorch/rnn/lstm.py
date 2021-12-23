from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from snorkel.learning.pytorch.rnn.rnn_base import RNNBase
from snorkel.learning.pytorch.rnn.utils import SymbolTable
from snorkel.models import Candidate


class LSTM(RNNBase):
    
    def _build_model(self, embedding_dim=50, hidden_dim=50, num_layers=1, dropout=0.25, bidirectional=False, small_features=10,
                     word_dict=SymbolTable(), **kwargs):
        self.word_dict = word_dict
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.word_dict.len(), self.embedding_dim, padding_idx=0)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim,
                            num_layers=num_layers, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True
                            )

        self.output_layer = nn.Linear(hidden_dim * self.num_directions, small_features)#self.cardinality if self.cardinality > 2 else 1)
        self.small_features = nn.Linear(small_features, self.cardinality if self.cardinality > 2 else 1)
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def inner (self, X, hidden_state):
        seq_lengths = torch.zeros((X.size(0)), dtype=torch.long)
        for i in range(X.size(0)):
            for j in range(X.size(1)):
                if X[i, j] == 0:
                    seq_lengths[i] = j
                    break
                seq_lengths[i] = X.size(1)

        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        X = X[perm_idx, :]
        inv_perm_idx = torch.tensor([i for i, _ in sorted(enumerate(perm_idx), key=lambda idx: idx[1])], dtype=torch.long)

        encoded_X = self.embedding(X)
        encoded_X = pack_padded_sequence(encoded_X, seq_lengths, batch_first=True)
        _, (ht, _) = self.lstm(encoded_X, hidden_state)
        output = ht[-1] if self.num_directions == 1 else torch.cat((ht[0], ht[1]), dim=1)
        return output[inv_perm_idx, :]
    
    def forward(self, X, hidden_state):
#         seq_lengths = torch.zeros((X.size(0)), dtype=torch.long)
#         for i in range(X.size(0)):
#             for j in range(X.size(1)):
#                 if X[i, j] == 0:
#                     seq_lengths[i] = j
#                     break
#                 seq_lengths[i] = X.size(1)

#         seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
#         X = X[perm_idx, :]
#         inv_perm_idx = torch.tensor([i for i, _ in sorted(enumerate(perm_idx), key=lambda idx: idx[1])], dtype=torch.long)

#         encoded_X = self.embedding(X)
#         encoded_X = pack_padded_sequence(encoded_X, seq_lengths, batch_first=True)
#         _, (ht, _) = self.lstm(encoded_X, hidden_state)
#         output = ht[-1] if self.num_directions == 1 else torch.cat((ht[0], ht[1]), dim=1)

#         return self.output_layer(self.dropout_layer(output[inv_perm_idx, :]))
        return self.small_features(self.output_layer(self.dropout_layer(self.inner(X,hidden_state))))

    def feature_outputs(self, X, batch_size):
        n = len(X)
        if not batch_size:
            batch_size = len(X)
        
        if isinstance(X[0], Candidate):
            X = self._preprocess_data(X, extend=False)
        
        outputs = torch.Tensor([])
        
        for batch in range(0, n, batch_size):
            
            if batch_size > len(X[batch:batch+batch_size]):
                batch_size = len(X[batch:batch+batch_size])
    
            hidden_state = self.initialize_hidden_state(batch_size)
            max_batch_length = max(map(len, X[batch:batch+batch_size]))
            
            padded_X = torch.zeros((batch_size, max_batch_length), dtype=torch.long)
            for idx, seq in enumerate(X[batch:batch+batch_size]):
                # TODO: Don't instantiate tensor for each row
                padded_X[idx, :len(seq)] = torch.LongTensor(seq)

            #output = self.inner(padded_X, hidden_state)
            output = self.output_layer(self.dropout_layer(self.inner(padded_X, hidden_state)))
            # TODO: Does skipping the cat when there is only one batch speed things up significantly?
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs
    
    def initialize_hidden_state(self, batch_size):
        return (
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim)
        )
