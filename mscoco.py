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

# 1. アノテーション（テキスト）と画像（Train 2017の一部）を落とす
!wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
!unzip -q annotations_trainval2017.zip

# 2. 画像は11万枚落とすと時間がかかるので、検証用に validation (5,000枚) を使う
# CLIPのスクラッチ実装の検証（Loss減少の確認）には5,000枚を使う
!wget -q http://images.cocodataset.org/zips/val2017.zip
!unzip -q val2017.zip

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

img, caption = coco_check[0]
print(f"画像サイズ: {img.shape}") #画像サイズ: torch.Size([3, 224, 224])
print(f"キャプション: {caption}") #キャプション: ['A woman stands in the dining area at the table.', 'A room with chairs, a table, and a woman in it.', 'A woman standing in a kitchen by a window', 'A person standing at a table in a room.', 'A living area with a television and a table']

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

#OpenAIのCLIPトークナイザー、CLIPConfigを使用
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

coco_check_dataset = CocoDataset(coco_check, tokenizer=tokenizer)
train_loader = DataLoader(coco_check_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

model = CLIP(3, config)

#GPUに設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device=device)
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    #今回はバッチサイズが小さいため、学習率を小さく設定する
    lr=1e-4,
    weight_decay=0.1,
    betas=(0.9, 0.98),
    eps=1e-6
)

#学習
clip_training(100, optimizer, loss_fn, train_loader, model, device)
#Epoch: 1, Loss: 4.890561401844025
#Epoch: 2, Loss: 4.783298581838608
#Epoch: 3, Loss: 4.783004021644592
#Epoch: 4, Loss: 4.782793271541595
#Epoch: 5, Loss: 4.782918417453766
#Epoch: 6, Loss: 4.782820379734039
#Epoch: 7, Loss: 4.783060783147812
#Epoch: 8, Loss: 4.78293662071228
#Epoch: 9, Loss: 4.783118611574173
#Epoch: 10, Loss: 4.774392712116241
#Epoch: 11, Loss: 4.73795782327652
#Epoch: 12, Loss: 4.7040659308433534
#Epoch: 13, Loss: 4.686181348562241
#Epoch: 14, Loss: 4.675025480985641
#Epoch: 15, Loss: 4.626086232066155
#Epoch: 16, Loss: 4.609638023376465
#Epoch: 17, Loss: 4.58611244559288
#Epoch: 18, Loss: 4.592645627260208
#Epoch: 19, Loss: 4.554648584127426
#Epoch: 20, Loss: 4.533100593090057
#Epoch: 21, Loss: 4.540729427337647
#Epoch: 22, Loss: 4.467138105630875
#Epoch: 23, Loss: 4.4797833383083345
#Epoch: 24, Loss: 4.447931632399559
#Epoch: 25, Loss: 4.41915545463562
#Epoch: 26, Loss: 4.413081759214402
#Epoch: 27, Loss: 4.37133549451828
#Epoch: 28, Loss: 4.348676073551178
#Epoch: 29, Loss: 4.324285340309143
#Epoch: 30, Loss: 4.285634797811508
#Epoch: 31, Loss: 4.288970050215721
#Epoch: 32, Loss: 4.236832624673843
#Epoch: 33, Loss: 4.198546090722084
#Epoch: 34, Loss: 4.1928436905145645
#Epoch: 35, Loss: 4.156223532557488
#Epoch: 36, Loss: 4.14785595536232
#Epoch: 37, Loss: 4.0815363973379135
#Epoch: 38, Loss: 4.0467874825000765
#Epoch: 39, Loss: 4.043712440133095
#Epoch: 40, Loss: 3.9732643947005273
#Epoch: 41, Loss: 3.92792084813118
#Epoch: 42, Loss: 3.8962228894233704
#Epoch: 43, Loss: 3.838035933673382
#Epoch: 44, Loss: 3.82153405547142
#Epoch: 45, Loss: 3.7653705686330796
#Epoch: 46, Loss: 3.7580698102712633
#Epoch: 47, Loss: 3.6877600848674774
#Epoch: 48, Loss: 3.6460234999656675
#Epoch: 50, Loss: 3.554791182279587
#Epoch: 51, Loss: 3.551248785853386
#Epoch: 52, Loss: 3.4778343379497527
#Epoch: 53, Loss: 3.442423462867737
#Epoch: 54, Loss: 3.408788961172104
#Epoch: 55, Loss: 3.3465714782476423
#Epoch: 56, Loss: 3.319806832075119
#Epoch: 57, Loss: 3.2943897634744643
#Epoch: 58, Loss: 3.23397576212883
#Epoch: 59, Loss: 3.1685674622654916
#Epoch: 60, Loss: 3.142235779762268
#Epoch: 61, Loss: 3.096373315155506
#Epoch: 62, Loss: 3.0266988635063172
#Epoch: 63, Loss: 3.0214655458927155
#Epoch: 64, Loss: 2.959819579869509
#Epoch: 65, Loss: 2.97844285517931
#Epoch: 66, Loss: 2.9399761885404585
#Epoch: 67, Loss: 2.871809098124504
#Epoch: 68, Loss: 2.8390878036618235
#Epoch: 69, Loss: 2.7882578313350677
#Epoch: 70, Loss: 2.757362684607506
#Epoch: 71, Loss: 2.745253840088844
#Epoch: 72, Loss: 2.6809787467122077
#Epoch: 73, Loss: 2.655967243015766
#Epoch: 74, Loss: 2.6428023397922518
#Epoch: 75, Loss: 2.62912005931139
#Epoch: 76, Loss: 2.5931675374507903
#Epoch: 77, Loss: 2.5624993331730366
#Epoch: 78, Loss: 2.543476539850235
#Epoch: 79, Loss: 2.5327521055936812
#Epoch: 80, Loss: 2.52274145334959
#Epoch: 81, Loss: 2.4971819519996643
#Epoch: 82, Loss: 2.4724835366010667
#Epoch: 83, Loss: 2.457848757505417
#Epoch: 84, Loss: 2.4389041140675545
#Epoch: 85, Loss: 2.3999921515583993
#Epoch: 86, Loss: 2.419611093401909
#Epoch: 87, Loss: 2.399204784631729
#Epoch: 88, Loss: 2.368404212594032
#Epoch: 89, Loss: 2.3651057571172713
#Epoch: 90, Loss: 2.360833888500929
#Epoch: 91, Loss: 2.3627604633569717
#Epoch: 92, Loss: 2.3227290511131287
#Epoch: 93, Loss: 2.346429693698883
#Epoch: 94, Loss: 2.3370402693748473
#Epoch: 95, Loss: 2.344551381468773
#Epoch: 96, Loss: 2.344741028547287
#Epoch: 97, Loss: 2.350475913286209
#Epoch: 98, Loss: 2.33016344755888
#Epoch: 99, Loss: 2.3197243474423885
#Epoch: 100, Loss: 2.3292756229639053
