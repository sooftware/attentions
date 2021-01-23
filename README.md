<p  align="center"><img src="https://user-images.githubusercontent.com/42150335/105606759-8dea5680-5dde-11eb-96a0-78f632a926c2.png" height=100>
  
<p  align="center">An Apache 2.0 PyTorch implementation of some attentions for Deep Learning Researchers.
  
***
  
<p  align="center"> 
     <a href="https://www.codefactor.io/repository/github/sooftware/attentions">
          <img src="https://www.codefactor.io/repository/github/sooftware/attentions/badge"> 
     </a>
     <a href="https://github.com/sooftware/KoSpeech/blob/latest/LICENSE">
          <img src="http://img.shields.io/badge/license-Apache--2.0-informational"> 
     </a>
     <a href="https://github.com/pytorch/pytorch">
          <img src="http://img.shields.io/badge/framework-PyTorch-informational"> 
     </a>
     <a href="https://www.python.org/dev/peps/pep-0008/">
          <img src="http://img.shields.io/badge/codestyle-PEP--8-informational"> 
     </a>
  
  
## Intro
  
`attentions` provides some attentions used in natural language processing using pytorch.   
these attentions can used in neural machine translation, speech recognition, image captioning etc...  
  
![image](https://user-images.githubusercontent.com/42150335/83331902-7bf9f780-a2d3-11ea-8f7e-172f55deef45.png)
  
`attention` allows to attend to different parts of the source sentence at each step of the output generation.   
Instead of encoding the input sequence into a single fixed context vector, we let the model learn how to generate a context vector for each output time step.  
  
## Implementation list
  
 
|Name|Citation|  
|---|---|  
|Additive Attention|[Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf)|  
|Dot-Product Attention|[Luong et al., 2015](https://arxiv.org/pdf/1508.04025.pdf)|  
|Location-Aware (Location Sensitive) Attention|[Chorowski et al., 2015](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf)|    
|Scaled Dot-Product Attention|[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)|  
|Multi-Head Attention|[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)|
|Relative Multi-Head Self Attention|[ZihangDai et al., 2019](https://arxiv.org/abs/1901.02860)|  
  
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/nlp-attentions/issues) on Github.  
or Contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com

