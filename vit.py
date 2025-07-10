import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


# Building Vision transformer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels=channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        n_patches = (img_size // patch_size) **2 # define the number of patches 
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1+n_patches, embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x) # (Batch, Embed, Height/Patch_size, Width/Patch_size)
        x = x.flatten(2).transpose(1,2) # (Batch, N_patch, Embed) 
        # flatten from the dimension 2 onward, switch dim2 and dim2 
        # because transformer expect (Batch, Seq_len, Embed_size)
        cls_token = self.cls_token.expand(B, -1, -1) 
        # Give me B copies of the CLS token -> (Batch, 1, Embed_size)
        x = torch.cat((cls_token, x), dim=1) # -> (Batch, 1 + N_patch, Embed_size) 
        x = x + self.pos_embed

        return x 
    
class MLP(nn.Module):
    def  __init__(self, in_features, hidden_dim, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features= in_features)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))  # F to replace nn.
        x = self.dropout(self.fc2(x))

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_dim, drop_rate):
        super().__init__()
        self.normalization1 = nn.LayerNorm(embed_dim) # applied per token, normalize the embedding dimension
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, drop_rate, batch_first= True)
        self.normalization2= nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x):
        attention, _ = self.attention(self.normalization1(x), self.normalization1(x), self.normalization1(x))  
        x = x + attention 
        x = x + self.mlp(self.normalization2(x)) 
        return x 
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, channels, n_classes, embed_dim, depth, n_heads, mlp_dim, drop_rate):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, channels, embed_dim)
        self.encoder = nn.Sequential(*[
            TransformerEncoder(embed_dim, n_heads, mlp_dim, drop_rate)
            for _ in range(depth) 
            ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]

        return self.head(cls_token)

def train(model, loader, optimizer, criterion, device):
    model.train()

    total_loss, correct = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # forward pass: model output raw logit
        out = model(x) # -> (examples in a batch, logit for all classes)
        # calculate the mean loss per batch (per sample)
        loss = criterion(out, y) # -> a scalar, mean loss for that batch 
        # backpropagation
        loss.backward()   
        # gradient descent
        optimizer.step()

        total_loss += loss.item() * x.size(0)  # mean loss per sample * batch size = total loss for this batch
        correct += (out.argmax(1) == y).sum().item() #  finds the index of the max value along dimension 1
    return total_loss / len(loader.dataset), correct / len(loader.dataset) # averge loss per sample & accuracy

# Evaluate model
def eval(model, loader, device):
    model.eval()
    correct = 0 

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
    return correct / len(loader.dataset)