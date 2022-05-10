# Reinforcement Learning for Dialogue Generation with emotion reward

**This is the repo for ECEN 689 Reinforcement Learning Project.**

We reimplement the [paper](https://arxiv.org/pdf/1606.01541.pdf) _Deep Reinforcement Learning for Dialogue Generation_ in PyTorch. Compared to the original paper, we changed the dataset.

Authors: Yuming Han, Ta-Wei Yeh. [Report](https://drive.google.com/file/d/1R1tH5sryEJTnfevhpnNkB-Br1WJPzHK3/view?usp=sharing).

## Abstract

> 

To overcome the lack of efficient of Seq2seq-based model on preprocessing long series horizons of information in RL, we explored the feasibility of stabilizing transformer-based network on deep reinforcement learning dialogue generation, the contributions of the present work are:

1. We improved the reward engineering procedure to obtain optimized responses on dialogue generators.

2. We evaluated the baseline on different model, ablation and data.

3. We explored the feasibility of using novel language models integrating RL training approaches to obtain better language generation.  

>

## Installation

* Clone this repo
* Install dependencies. We use python 3.7.4, pytorch >= 1.6.0 and cuda >= 10.1

```
pip install torch==1.6.0 torchvision==0.7.0
```

## Dataset

High-quality first sentence as the starting of dialogue is important to the simulation. Some sentences, such as "What?", are not good enough for the simulation. Those are so vague and lack of context, making it confusing to answer. To avoid this, we choose a very standard dataset named "**DailyDialog**". **DailyDialog** is a high-quality multi-turn dialog dataset, human-written and less noisy. Not only the dialogue text, it also has labels of communication intention and emotion information. However, in our experiments, we only use the texts.

Apart from this, we also use the scripts from **Lisen Dai, Xiangcong Kong, Yixian Cheng, Rui Sun** to train our model. [Repo](https://github.com/Ls-Dai/Deep-Reinforcement-Learning-for-Dialogue-Generation-in-PyTorch). We found the dataset from their repo. Or you may download the dataset [here](http://yanran.li/dailydialog). 

## Training

We trained the model on one NVIDIA 32GB V100 GPU. Training on GPUs is recommended.

```
python Sentiment_Classifier.py  # train emotion classifier
python train_seq2seq.py         # train Seq2Seq model
python rl.py                    # train RL with or without pre-trained seq2seq model
python naive_run.py             # to evaulate if RL learn emotion responses, or chat with the robot
```

## Experimental Results

![image](https://github.com/TaWeiYeh/Emotional-Dialogue/tree/main/images/result.png)

## Acknowledgement

Thanks for the repository from [Repo](https://github.com/Ls-Dai/Deep-Reinforcement-Learning-for-Dialogue-Generation-in-PyTorch) and [Repo](https://github.com/GameDisplayer/DRL4DG). 
