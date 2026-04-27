import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt


#画像埋め込み
class VisionEmbedding(nn.Module):
    def __init__(self, in_channel, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        #224×224の画像を16×16の画像に分けることを想定
        self.conv = nn.Conv2d(in_channel, self.embed_dim, kernel_size=16, stride=16)
        self.pos = nn.Parameter(torch.randn(1, 14*14 + 1, self.embed_dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, imgs):
        #(batch_size, embed_dim, 14, 14)にする
        imgs_divided = self.conv(imgs)

        #(B, patch_size, embed_dim)の順にする
        imgs_organize = rearrange(imgs_divided, 'b c h w -> b (h w) c')

        #クラストークン
        cls_token = self.cls.expand(imgs.size(0), -1, -1)
        imgs_embed = torch.cat((cls_token, imgs_organize), dim=1)

        #位置エンコーディング
        pos_embed = self.pos.expand(imgs.size(0), -1, -1)
        imgs_embed = imgs_embed + pos_embed

        return imgs_embed


#マルチヘッドアテンション
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.q_net = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_net = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_net = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_state):
        # 線形変換
        q = self.q_net(hidden_state)
        k = self.k_net(hidden_state)
        v = self.v_net(hidden_state)

        # (b, s, 512) -> (b, s, 8, 64) -> (b, 8, s, 64) に軸を入れ替える
        q = rearrange(q, 'b s (n d) -> b n s d', n=self.num_heads)
        k = rearrange(k, 'b s (n d) -> b n s d', n=self.num_heads)
        v = rearrange(v, 'b s (n d) -> b n s d', n=self.num_heads)

        # アテンション計算
        # (b, 8, s, 64) @ (b, 8, 64, s) -> (b, 8, s, s)
        scores = (q @ k.transpose(-2, -1)) / sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)

        # (b, 8, s, 64) -> (b, s, 512)
        out = weights @ v
        out = rearrange(out, 'b n s d -> b s (n d)')

        x = self.out_proj(out)
        return x


#Feed Forward層
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


#Transformer Encoder
class VisionTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer = nn.LayerNorm(self.embed_dim)
        self.multi = MultiHeadAttention(config)
        self.feed = FeedForward(config)

    def forward(self, hidden_state):
        #残差接続
        x = hidden_state + self.multi(self.layer(hidden_state))
        x = x + self.feed(self.layer(x))
        return x


#ViT
class VisionTransformer(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.embedding = VisionEmbedding(in_channels, config)
        self.transformer = VisionTransformerEncoder(config)
        self.mlp = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, n):
        #xは224×224を想定
        x = self.embedding(x)
        for _ in range(n):
            x = self.transformer(x)
        #クラスのテンソルのみを抽出
        x_class = x[:, 0]
        output = self.mlp(x_class).squeeze(1)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size,
                                             config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                 config.hidden_size)
        self.eos = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        #input_idsは(B, s)の形状を想定
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        #token_embeddings、position_embeddingsはともに(B, sequence, embed_dim)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddigns = token_embeddings + position_embeddings

        #eosを最後につける
        eos = self.eos.expand(input_ids.size(0), -1, -1)
        embeddings = torch.cat((embeddings, eos), dim=1)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


#テキストエンコーダ
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.encoder = TransformerEncoderLayer(config)
        self.mlp = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, n, eos_indices):
        x = self.embedding(x)
        for _ in range(n):
            x = self.encoder(x)

        #eosのみ抽出
        #xは(B, sequence + 1, embed_dim)
        x_eos = x[torch.arange(x.size(0)), eos_indices]
        output = self.mlp(x_eos).squeeze(1)

        return output


#CLIP
class CLIP(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.vit = VisionTransformer(in_channels, config)
        self.transformer = Transformer(config)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, imgs, texts, n, eos_indices):
        #学習の1ループ
        texts = self.transformer(texts, n, eos_indices)
        imgs = self.vit(imgs, n)
        #正規化
        texts_norm = F.normalize(texts, p=2, dim=1)
        imgs_norm = F.normalize(imgs, p=2, dim=1)

        #texts_normは(embed_dim, b), imgs_normは(b, embed_dim)
        #logitsは(b, b)
        scale = self.logit_scale.exp()
        logits = torch.matmul(imgs_norm, texts_norm.T) * scale

        #ラベルを作る
        labels = torch.arange(imgs.size(0), device=device)

        return logits, labels


#CLIPの検証
class CLIPClassifier(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.transformer = Transformer(config)
        self.vit = VisionTransformer(in_channels, config)

    def forward(self, img, texts, device):
        self.vit.eval()
        self.transformer.eval()
        with torch.no_grad():
            #GPUに移し、エンコード
            #imgは(C, H, W)を想定のため、unsqueezeでBの次元を増やす
            #textsはすでに投入できる物を想定のため、(B, sequence)を想定
            img = img.unsqueeze(0).to(device=device)
            texts = texts.to(device=device)
            img = self.vit(img)
            texts = self.transformer(texts)

            #imgは(1, embed_dim), textsは(1, embed_dim)
            #正規化
            texts_norm = F.normalize(texts, p=2, dim=1)
            img_norm = F.normalize(img, p=2, dim=1)

            #texts_normは(b, embed_dim), imgs_normは(1, embed_dim)
            #logitsは(b, 1)
            logits = torch.matmul(img_norm, texts_norm.T)

            #logitsから最大値のインデックスを取得
            max_index = torch.argmax(logits, dim=1)
            print(texts[max_index])

    
