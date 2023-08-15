import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import json

# hyperparameters

with open(f"{os.path.dirname(os.path.abspath(__file__))}\hyperparameters.json", "r") as file:
    hyperparameters = json.load(file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------

torch.manual_seed(1337)

with open(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}\\data set\\all_artist_data_artist.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

hyperparameters["stoi"] = stoi
hyperparameters["itos"] = itos
hyperparameters["vocab_size"] = vocab_size

print(str(len(text)) + "o text")
print(str(len(chars)) + " chars")

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - hyperparameters["block_size"], (hyperparameters["batch_size"],))
    x = torch.stack([data[i:i+hyperparameters["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+hyperparameters["block_size"]+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hyperparameters["eval_iters"])
        for k in range(hyperparameters["eval_iters"]):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(hyperparameters["n_embd"], head_size, bias=False)
        self.query = nn.Linear(hyperparameters["n_embd"], head_size, bias=False)
        self.value = nn.Linear(hyperparameters["n_embd"], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(hyperparameters["block_size"], hyperparameters["block_size"])))

        self.dropout = nn.Dropout(hyperparameters["dropout"])

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(hyperparameters["n_embd"], hyperparameters["n_embd"])
        self.dropout = nn.Dropout(hyperparameters["dropout"])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(hyperparameters["dropout"]),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # hyperparameters["n_embd"]: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, hyperparameters["n_embd"])
        self.position_embedding_table = nn.Embedding(hyperparameters["block_size"], hyperparameters["n_embd"])
        self.blocks = nn.Sequential(*[Block(hyperparameters["n_embd"], n_head=hyperparameters["n_head"]) for _ in range(hyperparameters["n_layer"])])
        self.ln_f = nn.LayerNorm(hyperparameters["n_embd"]) # final layer norm
        self.lm_head = nn.Linear(hyperparameters["n_embd"], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)

        return logits, loss
 
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last hyperparameters["block_size"] tokens
            idx_cond = idx[:, -hyperparameters["block_size"]:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
            print(decode(idx[0].tolist()))
            
        return idx
    
model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["learning_rate"])

for iter in range(hyperparameters["max_iters"]):

    # every once in a while evaluate the loss on train and val sets
    if iter % hyperparameters["eval_interval"] == 0 or iter == hyperparameters["max_iters"] - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
torch.save(model.state_dict(), f"{os.path.dirname(os.path.abspath(__file__))}\IA_parameters.pth")

with open(f"{os.path.dirname(os.path.abspath(__file__))}\hyperparameters.json", "w") as fichier:
    json.dump(hyperparameters, fichier, indent=4)