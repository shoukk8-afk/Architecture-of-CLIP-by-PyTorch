import torch


#CLIPの訓練ループ
def clip_training(n_epochs, optimizer, loss_fn, dataloader, model, n, eos_indices, device):
    model.train()
    for epoch in range(1, n_epochs + 1):
        total = 0
        loss_train = 0
        #imgsは(b, c, h, w), textsは(b, sequence)
        for imgs, texts in dataloader:
            #GPUに移す
            imgs = imgs.to(device=device)
            texts = texts.to(device=device)
            #内積、ラベルの取得
            logits, labels = model(imgs, texts, n, eos_indices)

            #画像方向、テキスト方向の損失をそれぞれ算出して平均を取る
            #論文ではdimを指定しているが、torchの損失関数はdimを指定できないため、転置で対処
            loss_i = loss_fn(logits, labels)
            loss_t = loss_fn(logits.T, labels)
            loss = (loss_i + loss_t) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #1回のループの損失を加算して訓練データの1エボックの損失の合計を出す
            loss_train += loss.item()

    print(f"Epoch: {epoch}, Loss: {loss_train / len(dataloader)}")