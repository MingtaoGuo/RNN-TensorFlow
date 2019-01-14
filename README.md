# RNN-TensorFlow
Some interesting applications of RNN, e.g. char rnn (pomes generation), seq2seq (machine translation), image captioning (NIC)


# Chinese tang poem generation
The method is shown in follow figure.
![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/poem.jpg)
# Results
|藏头诗1|藏头诗2|
|-|-|
|相逢不可见，不见不相逢。不见青山里，何人识所逢。|猪岖江畔尖，暮雨到山中。雨滴胶帆动，云深雁雁啼。|
|约郭不相见，空山不见君。不知何处去，不见故人心。|年少不相见，何人更盍簪。不知春草色，不见白云期。|
|中禁闻清洛，青山独自然。青山连水阔，白日向云多。|大堤春色晚，一概华輈春。不见春风里，何人更有情。|
|华堂临巨壑，千里类宸祠。警跸方回首，缧囚不可忘。|吉垣入天子，不见白云中。日暮无人到，山中有旧梯。|
|腾庐不可见，不见不相逢。不见青山里，何人识所逢。||
|飞草初生柳，春风不见君。青山无限路，白发不相关。||
|时无垠兮复，不是不相宜。||

# Machine translation
The method is shown in follow figure.
![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/seq2seq.jpg)
# Results
|English|->|Chinese|
|-|-|-|
|I want to go to school.|->|我想去上學。|
|I am here.|->|我在這裡。|
|You are here.|->|你在這裡。|
|Where are you?|->|你在哪儿?|
|What do you like?|->|你喜欢什么？|

# Image captioning
The method is shown in follow figure.
![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/nic.jpg)
# Results
|a dog is running through the grass .|a woman in a bikini is sitting on a rock in the water . |a snowboarder is jumping off a snowy hill . |a baseball player in a white uniform is about to throw a ball . |
|-|-|-|-|
|![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/1.jpg)|![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/2.jpg)|![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/3.jpg)|![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/4.jpg)|

|a young girl in a pink bathing suit is swimming underwater . |a man in a wetsuit surfs on a wave . |a black dog is running through the water . |a group of people are sitting around a table with a large table of food . |
|-|-|-|-|
|![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/5.jpg)|![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/6.jpg)|![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/7.jpg)|![](https://github.com/MingtaoGuo/RNN-TensorFlow/blob/master/IMGS/8.jpg)|

These results are under training about 40,000 iterations.

# Dataset
1. tang poems dataset: [https://github.com/todototry/AncientChinesePoemsDB](https://github.com/todototry/AncientChinesePoemsDB)
2. ENG2CHN dataset: [http://www.manythings.org/anki/](http://www.manythings.org/anki/)
3. flickr30k: [https://pan.baidu.com/s/1QAZq22mGJVMfGh0-rLiwxQ](https://pan.baidu.com/s/1QAZq22mGJVMfGh0-rLiwxQ)
```
├── ImageCaption
    ├── flickr30k
        ├── imgs
            ├── 36979.jpg
            ├── 65567.jpg
            ├── 81641.jpg
                 ...
        ├── results_20130124.token
```

##### I have uploaded the dataset to the repository except the flickr30k(too big :joy:).


# Requirements
1. python 3.5
2. tensorflow 1.4.0
3. pillow
4. numpy
5. pandas

# Reference
[1]. [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[2]. [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[3]. Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[C]//Advances in neural information processing systems. 2014: 3104-3112.

[4]. Vinyals O, Toshev A, Bengio S, et al. Show and tell: Lessons learned from the 2015 mscoco image captioning challenge[J]. IEEE transactions on pattern analysis and machine intelligence, 2017, 39(4): 652-663.







