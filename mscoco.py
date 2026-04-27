# %%
from google.colab import drive
import sys
import os

#ドライブをマウント
drive.mount('/content/drive')
# My Driveのパスをシステムパスに追加します
my_drive_path = '/content/drive/MyDrive'
if my_drive_path not in sys.path:
    sys.path.append(my_drive_path)

print("sys.path に My Drive を追加しました。")

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset
import random
from transformers import CLIPTokenizer, CLIPConfig
from CLIP.model import CLIP, CLIPClassifier
from CLIP.engine import clip_training

# %%
# 1. アノテーション（テキスト）と画像（Train 2017の一部）を落とす
!wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
!unzip -q annotations_trainval2017.zip

# 2. 画像は11万枚落とすと時間がかかるので、検証用に validation (5,000枚) を使う
# CLIPのスクラッチ実装の検証（Loss減少の確認）には5,000枚を使う
!wget -q http://images.cocodataset.org/zips/val2017.zip
!unzip -q val2017.zip

# %%
#自作CLIP用に224×224にリサイズ
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # CLIPの事前学習済み重み（ResNet/ViT）の多くがImageNetで学習されているため、
    # その統計値に合わせてNormalizeを行う。
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#自作CLIPの動作確認用
coco_check = datasets.CocoCaptions(
    root = 'val2017',
    annFile = 'annotations/captions_val2017.json',
    transform = transform
)

# %%
img, caption = coco_check[0]
print(f"画像サイズ: {img.shape}")
print(f"キャプション: {caption}")

# %%
#キャプションを5つの中から1つ選び、トークン化するDataLoaderの作成
class CocoDataset(Dataset):
    def __init__(self, coco_dataset, tokenizer=None):
        self.coco = coco_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, n):
        img, captions = self.coco[n]

        #キャプションをランダムに選ぶ
        caption = random.choice(captions)

        #トークナイザーがある場合はトークン化
        if self.tokenizer:
            caption = "picture of" + caption
            caption = self.tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors='pt')
            return img, caption.input_ids.squeeze(0)
        return img, caption

# %%
#OpenAIのCLIPトークナイザー、CLIPConfigを使用
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

# %%
coco_check_dataset = CocoDataset(coco_check, tokenizer=tokenizer)
train_loader = DataLoader(coco_check_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

# %%
model = CLIP(3, config)

# %%
#GPUに設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device=device)
loss_fn = nn.CrossEntropyLoss()

# %%
optimizer = optim.AdamW(
    model.parameters(),
    #今回はバッチサイズが小さいため、学習率を小さく設定する
    lr=1e-4,
    weight_decay=0.1,
    betas=(0.9, 0.98),
    eps=1e-6
)

# %%
clip_training(100, optimizer, loss_fn, train_loader, model, device)


