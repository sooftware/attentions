# :stuck_out_tongue_winking_eye: nlp-attentions
  
### Pytorch implementation of some attentions used in Natural Language Processing
  
[<img src="https://github.com/gentaiscool/end2end-asr-pytorch/raw/master/img/pytorch-logo-dark.png" height=18>](https://pytorch.org/) <img src="https://img.shields.io/badge/License-MIT-yellow" height=20>
  
## Intro
  
`nlp-attentions` provides some attentions used in natural language processing using pytorch.   
these attentions can used in neural machine translation, speech recognition, image captioning etc...  
  
![image](https://user-images.githubusercontent.com/42150335/83331902-7bf9f780-a2d3-11ea-8f7e-172f55deef45.png)
  
`Attention mechanism` allows to attend to different parts of the source sentence at each step of the output generation.   
Instead of encoding the input sequence into a single fixed context vector, we let the model learn how to generate a context vector for each output time step.  
  
## Implementation list
  
|Name|Alignment score function|Citation|  
|---|---|---|  
|Content-base|score(***s_t***, ***h_i***) = cosine\[***s_t***, ***h_i***\] |[Graves 2014](https://arxiv.org/abs/1410.5401)|  
|Additive|score(***s_t***, ***h_i***) = **v** tanh(**W**\[***s_t***;***h_i***\])|[Bahdanau 2015](https://arxiv.org/pdf/1409.0473.pdf)|  
|Dot-Product|score(***s_t***, ***h_i***) = ***s_t*** · ***h_i***|[Luong 2015](https://arxiv.org/pdf/1508.04025.pdf)|  
|Location-Aware|score(***s_t***, ***h_i***) = **w** tanh(**W*****s_t*** + **V*****h_i*** + ***b***)|[Chorowski 2015](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf)|    
|Multi-headed Location-Aware|Multi-head + Location-aware|-|  
|Scaled Dot-Product|score(***s_t***, ***h_i***) = ***s_t*** · ***h_i*** / **d_k**|[Vaswani 2017](https://arxiv.org/abs/1706.03762)|  
|Multi-Head|score(***Q***, ***K***, ***V***) = (head_1, ..., head_n) **W**|[Vaswani 2017](https://arxiv.org/abs/1706.03762)|  
   
## How To Use

* `Multi-headed Location-aware`
```python
B, L, H, T = 32, 3, 512, 131  # batch, num_layers, hidden_dim, seq_len
N_HEAD, N_CONV_OUT = 8, 10
attn = None

attention = MultiHeadedLocationAwareAttention(H, N_HEAD, N_CONV_OUT)

# examples
input_var = torch.FloatTensor(B, 1, H)
hidden = torch.zeros(L, B, H)
value = torch.FloatTensor(B, T, H)

query, hidden = nn.GRU(input_var, hidden)
output, attn = attention(query, value, attn)
```

* `Location-aware` 
```python
B, L, H, T = 32, 3, 512, 131  # batch, num_layers, hidden_dim, seq_len
N_HEAD, N_CONV_OUT, ATTN_DIM = 8, 10, 256
attn = None

attention = LocationAwareAttention(H, ATTN_DIM, N_CONV_OUT, smoothing=True)

# examples
input_var = torch.FloatTensor(B, 1, H)
hidden = torch.zeros(L, B, H)
value = torch.FloatTensor(B, T, H)

query, hidden = nn.GRU(input_var, hidden)
output, attn = attention(query, value, attn)
```

* `Multi-head`
```python
B, L, H, T = 32, 3, 512, 131  # batch, num_layers, hidden_dim, seq_len
N_HEAD = 8

attention = MultiHeadAttention(H, N_HEAD)

# examples
input_var = torch.FloatTensor(B, 1, H)
hidden = torch.zeros(L, B, H)
value = torch.FloatTensor(B, T, H)

query, hidden = nn.GRU(input_var, hidden)
output = attention(query, value)
```
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/nlp-attentions/issues) on Github.  
or Contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## Citing
```
@github{
  title = {nlp-attentions},
  author = {Soohwan Kim},
  publisher = {GitHub},
  url = {https://github.com/sooftware/nlp-attentions},
  year = {2020}
}
```
