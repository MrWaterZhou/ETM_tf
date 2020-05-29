## original pytorch implement: https://github.com/adjidieng/ETM

## Dependencies

+ python 3.7
+ tensorflow 2.1.0

## Preprocess Data

1. tokenize text and generate vocabulary file(vocab.txt, first line must be "\<PAD\>")
2. encode text as ids(data.txt, a sentence per line, you don't have to pad each line to a same length)
3. train word2vec(embeddings.txt, you can use code from the original implement https://github.com/adjidieng/ETM)

## To Run

To learn interpretable embeddings and topics :
```
python train.py --data_path data.txt --batch_size 512 --vocab_path vocab.txt --train_embeddings 1 --lr 0.0005 --epochs 1000
```


To learn interpretable topics using ETM with pre-fitted word embeddings :

```
python train.py --data_path data.txt --batch_size 512 --vocab_path vocab.txt --train_embeddings 0 --lr 0.0005 --epochs 1000 --emb_path embeddings.txt```
```

## Some Changes

1. Using sequence of word ids as model input, easier to pre-process, but may cost more video memory(depend on the lenth of input)  

2. Using DenseEncoder instead of original BOW encoder(mathematically equivalent), since you may encounter with nan weight using sparse matrix

3. Metrics like ppl are not implemented 

4. Predefined topic words(may help to get reasonable topics)

## Citation

```
@article{dieng2019topic,
  title={Topic modeling in embedding spaces},
  author={Dieng, Adji B and Ruiz, Francisco J R and Blei, David M},
  journal={arXiv preprint arXiv:1907.04907},
  year={2019}
}
```

