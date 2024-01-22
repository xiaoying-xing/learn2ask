import torch.nn as nn
import torch.nn.functional as F
import torch


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, word_embeddings, n_layers=1, dropout=0.1, update_wd_emb=False):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(len(word_embeddings), len(word_embeddings[0]))
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding.weight.requires_grad = update_wd_emb
        self.gru = nn.GRU(len(word_embeddings[0]), hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # outputs, hidden = self.gru(packed, hidden)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        # outputs of shape `(seq_len, batch, num_directions * hidden_size)`
        # hidden of shape `(num_layers * num_directions, batch, hidden_size)`
        outputs, hidden = self.gru(embedded, hidden)
        # (seq_len, batch, hidden_size)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        """

        :param hidden: [max_len, bs, hidden_size]
        :param encoder_outputs:  [max_len, bs, hidden_size]
        :return:
        """

        attn_energies = torch.bmm(hidden.transpose(0, 1), encoder_outputs.transpose(0, 1).transpose(1, 2)).squeeze(1)
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class GeneralAttn(nn.Module):
    def __init__(self, hidden_size):
        super(GeneralAttn, self).__init__()
        self.hidden_size = hidden_size
        self.weight = torch.randn((self.hidden_size, self.hidden_size))
        self.weight = self.weight.cuda()

    def forward(self, hidden, encoder_outputs):
        """

        :param hidden: [1, bs, hidden_size]
        :param encoder_outputs:  [max_len, bs, hidden_size]
        :return:
        """

        attn_energies = torch.bmm(torch.matmul(hidden.transpose(0, 1), self.weight),
                                  encoder_outputs.transpose(0, 1).transpose(1, 2)).squeeze(1)
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, word_embeddings, attn_type='dot', n_layers=1, dropout=0.1,
                 update_wd_emb=False, condition='replace'):
        super(AttnDecoderRNN, self).__init__()

        assert condition in {"none", "replace", "concat"}
        self.condition = condition
        self.hidden_size = hidden_size
        self.input_emb_size = 2 * len(word_embeddings[0]) if condition == "concat" else len(word_embeddings[0])
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(len(word_embeddings), len(word_embeddings[0]))
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding.weight.requires_grad = update_wd_emb
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.input_emb_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        if attn_type == "dot":
            self.attn = Attn(hidden_size)
        else:
            self.attn = GeneralAttn(hidden_size)

    def forward(self, input_seq, last_hidden, p_encoder_outputs, pre_embedded=None):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        if pre_embedded is None or self.condition == 'none':
            embedded = self.embedding(input_seq)
        elif self.condition == "replace":
            embedded = pre_embedded
        elif self.condition == "concat":
            embedded = self.embedding(input_seq)
            embedded = torch.cat((embedded, pre_embedded), dim=-1)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, embedded.shape[0], embedded.shape[1])  # S=1 x B x N
        # import pdb; pdb.set_trace()
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all p_encoder outputs;
        # apply to p_encoder outputs to get weighted average
        p_attn_weights = self.attn(rnn_output, p_encoder_outputs)
        p_context = p_attn_weights.bmm(p_encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N

        p_context = p_context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, p_context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden
