## PIRE: Adversarial Queries for Blocking Content-based Image Retrieval (CBIR).

This repository releases the pytorch implementation for PIRE that generates adversarial examples for neural feature-based CBIR.
Now PIRE only supports to generate adversarial queries for state-of-the-art CNN image retrieval method GeM[1]. 

GeM neural feature extraction code and pre-trained ResNet-101-GeM model refer to:
[cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch)

### Pytorch implementaiton of PIRE:
#### Prerequisites

Python3<br/>
PyTorch 1.0.0<br/>
GPU supported (Tested with Python 3.6.6 on Ubuntu 16.04)<br/>

#### How to use:

To get the PIRE (T = 500) results in our paper, please run:

```
python3 gen_pire.py -T "500" -gpu_id "0" -cnnmodel "gem" -in_dir "./img_input/" -out_dir "./img_output/" -p True
```


Detailed explanation of PIRE's parameters can be checked by:

```
python3 gen_pire.py -h
```

### Experimental results:
	
Examples of generated images are shown below:

![patches](https://github.com/liuzrcc/PIRE/blob/master/examples/PIRE_exp_1.jpg)



The decrease of mean average precision (mAP) in GeM is shown in this table:



|                  | Oxford5k (BB/WI)                   | Paris6k(BB/WI)                     |
|------------------|-----------------------------|-----------------------------|
| Original Queries | 78.39/74.42                 | 87.27/87.26                 |
| PIRE (T = 200)   | 22.98/18.00                 | 34.49/26.53                 |
| PIRE (T = 500)   | 3.93/2.31                   | 10.53/7.18                  |



It shows that adversarial image query generated with enough rounds (T=500) and fewer rounds (T=200) can both
strongly decreases the performance of neural-feature-based CBIR.



Some exaples of ranked list and calculated mean average precision (mAP). From these exampes, it is observed that PIRE strongly decrease the performance of neural feature-based CBIR.

![patches](https://github.com/liuzrcc/PIRE/blob/master/examples/PIRE_exp_2.jpg)





Please cite the following paper if you use PIRE in your research.

      @misc{pire2019,
      Author = {Zhuoran Liu and Zhengyu Zhao and Martha Larson},
      Title = {Who's Afraid of Adversarial Queries? The Impact of Image Modifications on Content-based Image Retrieval},
      Year = {2019},
      Eprint = {arXiv:1901.10332},
      }
      
The copyright of all the images belongs to the image owners.



## Reference
[1] Radenović, Filip, Giorgos Tolias, and Ondrej Chum. 
"Fine-tuning CNN image retrieval with no human annotation." IEEE Transactions on Pattern Analysis and Machine Intelligence (2018).
