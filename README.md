# Attention-Implementation
  
## Pytorch Implementation of Some Attention
  
[<img src="https://github.com/gentaiscool/end2end-asr-pytorch/raw/master/img/pytorch-logo-dark.png" height=18>](https://pytorch.org/) <img src="https://img.shields.io/badge/License-MIT-yellow" height=20>
  
## Implementation list
  
|Name|Alignment score function|Citation|  
|---|---|---|  
|Content-base|score(***s_t***, ***h_i***) = cosine\[***s_t***, ***h_i***\] |[Graves2014](https://arxiv.org/abs/1410.5401)|  
|Additive|score(***s_t***, ***h_i***) = **v** tanh(**W**\[***s_t***;***h_i***\])|[Bahdanau2015](https://arxiv.org/pdf/1409.0473.pdf)|  
|Dot-Product|score(***s_t***, ***h_i***) = ***s_t*** · ***h_i***|[Luong2015](https://arxiv.org/pdf/1508.04025.pdf)|  
|Hybrid|score(***s_t***, ***h_i***) = **w** tanh(**W*****s_t*** + **V*****h_i*** + ***b***)|[Chorowski2015](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf)|    
|Multi-Head|concat(head_1, ..., head_n) **W**|[Vaswani2017](https://arxiv.org/abs/1706.03762)|  

  
* Multi-Head Attention: [Citation](https://arxiv.org/abs/1712.01769)  
  
* Dot Product Attention: [Citation](https://arxiv.org/pdf/1508.04025.pdf)  
  
* Additive Attention: [Citation](https://arxiv.org/abs/1409.0473)
  
* Hybrid Attention: [Citation](https://arxiv.org/pdf/1506.07503.pdf)  
  
* Content-based Attention: [Citation](https://arxiv.org/pdf/1506.07503.pdf)  
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sh951011/Attention-Implementation/issues) on Github.  
or Contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## License
```
MIT License

Copyright (c) 2020 KimSooHwan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
