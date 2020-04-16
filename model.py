import torch
import torch.nn as nn
import torch.nn.functional as F

class HierAttnNet(nn.Module):
    def __init__(self,
                 doc_len,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 tagset_size,
                 default_dropout_rate=0.5,
                 embedding_pretrained=None,
                 embedding_freeze=True):
        super(HierAttnNet, self).__init__()

        self.doc_len = doc_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.softmax = nn.Softmax(dim=1)

        # word

        if embedding_pretrained is None:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.word_embeddings.weight.data.copy_(embedding_pretrained)
            if embedding_freeze:
                self.word_embeddings.requires_grad = False
        self.embeds_drop = nn.Dropout(default_dropout_rate)

        self.gru_word = nn.GRU(embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.gru_word_drop = nn.Dropout(default_dropout_rate)
        self.gru_word_hidden = None

        self.hidden2hidden_word = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.hidden2hidden_word_drop = nn.Dropout(default_dropout_rate)

        self.attn_word = torch.Tensor(2 * self.hidden_dim)
        self.attn_word.data.uniform_(-0.1, 0.1)


        # final

        self.v_drop = nn.Dropout(default_dropout_rate)

        self.doclen2tag = nn.Linear(2 * self.hidden_dim, tagset_size)

    def forward(self, doc):
        """ Forward.

        Args:
            sent: (batch_size, doc_len)
        """

        #####################
        # word
        #####################

        embeds = self.word_embeddings(doc)
        embeds = self.embeds_drop(embeds)
        # (batch_size, doc_len, embedding_dim)

        h_it, self.gru_word_hidden = self.gru_word(embeds, None)
        h_it = self.gru_word_drop(h_it)
        # (batch_size, doc_len, 2 * hidden_dim)

        alpha_it = torch.matmul(h_it, self.attn_word)
        # (batch_size, doc_len)

        alpha_it = self.softmax(alpha_it)
        # (batch_size, doc_len)

        att_weight = alpha_it
        # (batch_size, doc_len)

        alpha_it = alpha_it.unsqueeze(2).expand_as(h_it)
        # (batch_size, doc_len, 2 * hidden_dim)

        a_i = alpha_it * h_it
        # (batch_size, doc_len, 2 * hidden_dim)

        v = torch.sum(a_i, dim=1)
        # (batch_size, 2 * hidden_dim)


        #####################
        # final
        #####################

        v = self.v_drop(v)
        tag_space = self.doclen2tag(v)
        # (batch_size, tagset_size)

        tag_scores = F.log_softmax(tag_space, dim=1)
        # (batch_size, tagset_size)

        return tag_scores, att_weight
