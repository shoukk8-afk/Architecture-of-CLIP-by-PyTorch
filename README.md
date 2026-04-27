# Architecture-of-CLIP-by-PyTorch


OpenAIの[CLIP](https://openai.com/ja-JP/index/clip/)をPyTorchのみで実装しました。Transformersライブラリを使用せず、PyTorchのみでCLIPをスクラッチ実装した背景には、Transformerの挙動やViTの構造を深く理解したいという目的がありました。また、これまでの『書籍を参考にした実装』から脱却し、論文から直接実装に落とし込む経験を積むことも狙いとしています。


使用ライブラリ
* PyTorch
* einops
* math


##Project Structure


### 参考文献


* **Paper**

  
  * **CLIP**: [*Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)

    * CLIPのアーキテクチャ、損失の計算のNumPyでの疑似コードを参考にしました。

  * **ViT(Vision Transformer)**: [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
 
    * ViTの詳細なアーキテクチャを参考にViTの実装を行いました。

  * **Transformer / GPT**
 
    * [*Language Models are Unsupervised Multitask Learners* (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
     
    * [*Improving Language Understanding by Generative Pre-Training* (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
     
    * [*Attention Is All You Need* (Ashish et al.. 2017)](https://arxiv.org/abs/1706.03762)

      * テキストエンコーダの参考にしました。 
     
* **Books**


  * Lewis Tunstall, Leandro von Werra and Thomas Wolf, 『機械学習エンジニアのためのTransformers』, 中山光樹訳, オライリー･ジャパン, 2023


    * テキストエンコーダのコードは3章「Transformerの詳細」を参考にしました。


  * David Foster, 『生成Deep Learning』, 松田晃一, 小沼千絵訳, オライリー･ジャパン, 2024


    * CLIPの大まかなアーキテクチャを13章「マルチモーダルモデル」を参考にしました。
