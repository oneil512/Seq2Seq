import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import operator

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='float32')[y]

# Get data
f = open('/Users/a6002157/pytorchNet/shakespeare.txt')
timesteps = 100
device = torch.device("cpu")
MAX_LENGTH = 150
batch_size = 64
epochs = 1

attn_model = 'dot'
encoder_n_layers = 2
decoder_n_layers = 1
dropout = 0.1
checkpoint_iter = 4000
model_name='textgen'

xtrain_lines = []
l = f.readlines()
for line in l:
  xtrain_lines += list(line)
#xtrain_lines = xtrain_lines[:10000]

xtest_lines = xtrain_lines[1:] # Test data is shifted by one char

# Pad the data so it fits into timesteps
if len(xtrain_lines) % timesteps != 0:
  extra = timesteps - (len(xtrain_lines) % timesteps)
  xtrain_lines += [' '] * extra

extra = timesteps - (len(xtest_lines) % timesteps)
xtest_lines += [' '] * extra

xtrain_lines = np.array(xtrain_lines)
xtest_lines = np.array(xtest_lines)
classes = set(xtrain_lines)
classes = list(classes)
vocab_size = len(classes)

hidden_size = 256
#SOS_token = [0.0] * hidden_size
SOS_token = 0


cat_d = {key: value for (key, value) in list(zip(classes, range(len(classes))))}
r_cat_d = {value: key for (key, value) in list(zip(classes, range(len(classes))))}
#xtrain_lines = to_categorical(np.array(list(map(lambda x : cat_d[x], xtrain_lines))), vocab_size)
#xtest_lines = to_categorical(np.array(list(map(lambda x : cat_d[x], xtest_lines))), vocab_size)


xtrain_lines = np.array(list(map(lambda x : cat_d[x], xtrain_lines)))
xtest_lines = np.array(list(map(lambda x : cat_d[x], xtest_lines)))

#xtrain_lines = np.reshape(xtrain_lines, [-1, timesteps, len(classes)])
#xtest_lines = np.reshape(xtest_lines, [-1, timesteps, len(classes)])

xtrain_lines = np.reshape(xtrain_lines, [-1, timesteps])
xtest_lines = np.reshape(xtest_lines, [-1, timesteps])

num_batches = xtrain_lines.shape[0]

class Encoder(torch.nn.Module):
  def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.dropout = dropout
    self.embedding = embedding

    #input size is a word embedding with number of features = hidden_size
    self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=False)

  def forward(self, input_seq, hidden=None):
    embedded = self.embedding(input_seq)
    outputs, hidden = self.gru(embedded, hidden)
    # Sum bidirectional GRU outputs
    #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
    return outputs, hidden

class LuongAttn(torch.nn.Module):
  def __init__(self, method, hidden_size):
    super(LuongAttn, self).__init__()
    self.method = method
    if self.method not in ['dot', 'general', 'concat']:
      raise ValueError(self.method, "is not attention method")
    self.hidden_size = hidden_size
    if method == 'general':
      self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
    elif method == 'concat':
      self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
      self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

  def dot_score(self, hidden, encoder_output):
    return torch.sum(hidden * encoder_output, dim=2)

  def general_score(self, hidden, encoder_output):
    energy = self.attn(encoder_output)
    return torch.sum(hidden * energy, dim=2)

  def concat_score(self, hidden, encoder_output):
    energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
    return torch.sum(self.v * energy, dim=2)

  def forward(self, hidden, encoder_outputs):
    if self.method == 'general':
      attn_energies = self.general_score(hidden, encoder_outputs)
    if self.method == 'dot':
      attn_energies = self.dot_score(hidden, encoder_outputs)
    if self.method == 'concat':
      attn_energies = self.concat_score(hidden, encoder_outputs)

    return F.softmax(attn_energies.t(), dim=1).unsqueeze(1)
  
class LuongAttnDecoder(torch.nn.Module):
  def __init__(self, attn_model, hidden_size, output_size, embedding, n_layers=1, dropout=0.1):
    super(LuongAttnDecoder, self).__init__()

    self.attn_model = attn_model
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout = dropout
    self.embedding = embedding

    # Define layers
    self.dropout_layer = nn.Dropout(p=dropout)
    self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
    self.concat = nn.Linear(hidden_size * 2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

    self.attn = LuongAttn(attn_model, hidden_size)

  def forward(self, input_step, last_hidden, encoder_outputs):
    # Run this step one word at a time
    embedded = self.embedding(input_step)
    dropped_input = self.dropout_layer(embedded)

    rnn_output, hidden = self.gru(dropped_input, last_hidden)
    attn_weights = self.attn(rnn_output, encoder_outputs)
    context = attn_weights.bmm(encoder_outputs.transpose(0,1))

    # Get attentional hidden state h˜t = tanh(Wc[ct;ht])
    rnn_output = rnn_output.squeeze(0)
    context = context.squeeze(1)
    contextual_input = torch.cat((rnn_output, context), 1)
    attentional_hidden = torch.tanh(self.concat(contextual_input))

    output = self.out(attentional_hidden)
    output = F.softmax(output, dim=1)
    # Return output and final hidden state
    return output, hidden, attentional_hidden
  
def XELoss(pred, target):
  logPreds = torch.log(pred)
  xEntropies = torch.zeros((0), requires_grad=True, device=device)
  for i, p in enumerate(logPreds):
    j = -p.dot(target[i]).view(1)
    xEntropies = torch.cat((xEntropies, j), dim=0)
  xEntropy = torch.mean(xEntropies)
  return xEntropy

class SamplingSearchDecoder(nn.Module):
  def __init__(self, encoder, decoder):
    super(SamplingSearchDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input_seq, max_length):
    # Forward input through encoder model
    encoder_outputs, encoder_hidden = self.encoder(input_seq)
    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Initialize decoder input with SOS_token
    decoder_input = torch.Tensor(SOS_token).view(1,1,len(SOS_token))
    decoder_input.to(device)
    # Initialize tensors to append decoded words to
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    # Iteratively decode one word token at a time
    for _ in range(max_length):
        # Forward pass through decoder
        decoder_output, decoder_hidden, attentional_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
        # Obtain most likely word token and its softmax score
        prob_dist = torch.distributions.Categorical(decoder_output)
        decoder_input = prob_dist.sample()
        # Record token and score
        all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
        # Prepare current token to be next decoder input (add a dimension)
        decoder_input = tokenize(r_cat_d[decoder_input.item()])
    # Return collections of word tokens and scores
    return all_tokens

class BeamSearchDecoder(nn.Module):
  def __init__(self, encoder, decoder, beam_width):
    super(BeamSearchDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.beam_width = beam_width

  def forward(self, input_seq, max_length):
    # Forward input through encoder model
    encoder_outputs, encoder_hidden = self.encoder(input_seq)
    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Initialize decoder input with SOS_token
    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input.to(device)

    beams = []
    candidates = []

    # First pass to initialize the beam
    decoder_output, decoder_hidden, attentional_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
    scores, preds = decoder_output.topk(beam_width, dim=1)
    scores, preds = scores.view(self.beam_width), preds.view(self.beam_width)
    for i in range(self.beam_width):
      beams.append(([r_cat_d[preds[i].item()], scores[i].item()], decoder_hidden))

    while len(beams) > 0:
      beam = beams[0]
      decoder_input = torch.LongTensor([cat_d[beam[0][0][-1]]]).view(1,1)
      # Forward pass through decoder
      decoder_output, decoder_hidden, attentional_hidden = self.decoder(decoder_input, beam[1], encoder_outputs)
      scores, preds = decoder_output.topk(beam_width, dim=1)
      scores, preds = scores.view(self.beam_width), preds.view(self.beam_width)

      pairs = []
      for j in range(self.beam_width):
        pair = []
        pair.append(r_cat_d[preds[j].item()])
        pair.append(scores[j].item())
        pairs.append(pair)

      for p in pairs:
        prob = beam[0][1] * p[1]
        s = beam[0][0] + p[0]
        new_pair = [s, prob]
        if len(s) >= max_length:
          candidates.append(new_pair)
          continue
        beams.append((new_pair, decoder_hidden))

      beams.remove(beam)
      # Prune down to beam_width after 8 iterations
      if len(beams) % (self.beam_width ** 8) == 0 and self.beam_width > 1:
        beams.sort(key = lambda x: x[0][1], reverse=True)
        beams = beams[:50]
    candidates.sort(key = lambda x: x[1], reverse=True)
    return candidates

def train(input_variable, target_variable, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
  # Zero gradients
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  # Set device options
  input_variable = input_variable.to(device)
  target_variable = target_variable.to(device)

  # Initialize variables
  loss = 0
  print_losses = []
  n_totals = 0

  # Random initial hidden state, 2 for bidirectional
  #initial_hidden = torch.randn(2 * encoder.n_layers , batch_size, hidden_size)
  initial_hidden = torch.randn(encoder.n_layers , batch_size, hidden_size)
  encoder_outputs, encoder_hidden = encoder(input_variable, initial_hidden)
  
  # Create initial decoder input (start with SOS tokens for each sentence)
  #decoder_input = torch.Tensor([[SOS_token for _ in range(batch_size)]])
  decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
  decoder_input = decoder_input.to(device)

  decoder_hidden = encoder_hidden[:decoder.n_layers]
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
  # Forward batch of sequences through decoder one time step at a time
  if use_teacher_forcing:
      for t in range(max_target_len):
          decoder_output, decoder_hidden, attentional_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
          # Teacher forcing: next input is current target
          #decoder_input = target_variable[t].view(1, batch_size, vocab_size)

          decoder_input = target_variable[t].view(1, batch_size)
          decoder_input = decoder_input.to(device)
          # Calculate and accumulate loss
          xLoss = XELoss(decoder_output, torch.Tensor([to_categorical(x, vocab_size) for x in target_variable[t]]))
          loss += xLoss
          print_losses.append(xLoss)
  else:
      for t in range(max_target_len):
          decoder_output, decoder_hidden, attentional_hidden  = decoder(decoder_input, decoder_hidden, encoder_outputs)
          # Take most likely output across the batches
          preds_confidence, preds = decoder_output.topk(1, dim=1)
          decoder_input = torch.LongTensor([[preds[i][0] for i in range(batch_size)]])
          decoder_input = decoder_input.to(device)
          # Calculate and accumulate loss
          xLoss = XELoss(decoder_output, torch.Tensor([to_categorical(x, vocab_size) for x in target_variable[t]]))
          loss += xLoss
          print_losses.append(xLoss)

  # Perform backpropatation
  loss.backward()

  # Clip gradients: gradients are modified in place
  _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
  _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

  # Adjust model weights
  encoder_optimizer.step()
  decoder_optimizer.step()

  return sum(print_losses) / timesteps

def trainIters(epoch, model_name, train_lines, test_lines, encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_n_layers, decoder_n_layers, n_iteration, batch_size, print_every, clip):

    # Initializations
    print('Initializing')
    start_iteration = 1
    print_loss = 0
    learning_rate = .001

    # Training loop
    print("Training")
    for step, iteration in enumerate(range(start_iteration, n_iteration + 1)):

        input_variable = torch.tensor([next(train_lines) for _ in range(batch_size)])
        target_variable = torch.tensor([next(test_lines) for _ in range(batch_size)])

        #input_variable = np.transpose(input_variable, (1, 0, 2))
        #target_variable = np.transpose(target_variable, (1, 0, 2))

        input_variable = np.transpose(input_variable, (1, 0))
        target_variable = np.transpose(target_variable, (1, 0))

        max_target_len = timesteps

        #######TEST############
        if (step + 1) % 10000 == 0:
          learning_rate = learning_rate ^ 1.11
          encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
          decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * 5)

        # Run a training iteration with batch
        loss = train(input_variable, target_variable, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Epoch: {}; Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(epoch, iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

def evaluate(searcher, sentence, max_length=MAX_LENGTH):
    #tokens = tokenize(sentence)
    tokens = torch.LongTensor(list(map(lambda x : cat_d[x], sentence))).view(len(sentence), 1)
    tokens = tokens.to(device)
    # Decode sentence with searcher
    beams = searcher(tokens, max_length)
    # indexes -> words
    decoded_words = [beam[0] for beam in beams]
    return decoded_words

def tokenize(sentence):
  tokens = np.expand_dims(np.array(list(map(lambda x : to_categorical(cat_d[x], vocab_size), sentence))), axis=0)
  tokens = np.reshape(tokens, [1, -1, vocab_size])
  tokens = np.transpose(tokens, (1, 0, 2))

  return torch.Tensor(tokens)

def untokenize(tokens):
  sentence = list(map(lambda x : r_cat_d[x], tokens))

embedding = nn.Embedding(vocab_size, hidden_size)
encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoder(attn_model, hidden_size, vocab_size, embedding, decoder_n_layers, dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)
print('models built')

clip= 50.0
teacher_forcing_ratio = 1.0
decoder_learning_ratio = 5.0
n_iteration = num_batches // batch_size
print_every = 1
save_every = 500
beam_width = 2

encoder.train()
decoder.train()

searcher = BeamSearchDecoder(encoder, decoder, beam_width)

for epoch in range(epochs):
  learning_rate = .001 / (1 + epoch*100)
  encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

  train_lines = iter(xtrain_lines)
  test_lines = iter(xtest_lines)
  trainIters(epoch, model_name, train_lines, test_lines, encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_n_layers, decoder_n_layers, n_iteration, batch_size, print_every, clip)

sentences = ['To be or not to be ', 'For whom ', 'Do my eyes ', 'All that glitters is not ', 'Hell is empty and all the devils are ', 'To thine own self be true, and it must follow, as the night the day, thou canst ']
with open('res.txt', 'w+') as f:
  for sentence in sentences:
    words = evaluate(searcher, sentence, MAX_LENGTH)
    if str(type(searcher)) == "<class '__main__.BeamSearchDecoder'>":
      for word in words:
        f.write(sentence + word)
    else:
      f.write(sentence + word)
