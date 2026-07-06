# makemore

A Character-level language model built while working through Andrej Karpathy's "Neural Networks: Zero to Hero" series. Implements the makemore models. As of now, the character-level language model is implemented using bigram counting models only. This model was trained on a dataset of names to generate new, name-like words.


## Status

Currently bigram and MLP-based character model only. Next: RNN, GRU and Transformer-based character model (in progress).


## What's in here

- `bigram/bigram_using_tensors.ipynb` — count-based bigram model, implemented using tensor functions available in the pytorch library.
- `bigram/bigram_using_neural_nets.ipynb` — count-based bigram model, implemented using a simple single-layered neural network.
- `bigram/plot_tensor.ipynb` — plots the bigram model in the form of a (27,27) tensor
- `names.txt` — training data (32K names)
- `bigram/tensor.png` — the image of the tensor created in `plot_tensor.ipynb`
- `mlp/underfitted_mlp.ipynb` - an MLP-based model configured to underfit the training data; implemented using 6881 parameters. Following Bengio et al. 2003.
- `mlp/overfitted_mlp.ipynb` - an MLP-based model configured to overfit the training data; implemented using 26967 parameters. Following Bengio et al. 2003. 


## How it works

In the bigram implementation, the model learns the probability of the next character given the previous one, by counting co-occurences directly from the training data / learning them via gradient descent. Sampling from this probability distribution generates new names character by character until an end token is produced.

The MLP implementation extends the idea of using a wider context window, (multiple preceding characters instead of just one). Each character is mapped to a learned embedding vector. concatenated across the context window, and passed through a hidden layer to predict the next character. The underfitted and overfitted model uses 6881 and 269767 parameters respectively, showing how much the gap between the training and validation loss is affected by model capacity.

## Usage

```bash
git clone https://github.com/lalitadityamalik/makemore.git
cd makemore/bigram
jupyter notebook bigram_using_tensors.ipynb
jupyter notebook bigram_using_neural_nets.ipynb
jupyter notebook plot_tensor.ipynb

cd ../mlp
jupyter notebook underfitted_mlp.ipynb
jupyter notebook overfitted_mlp.ipynb
```


## Results

| Model              | Avg NLL (loss) |
| ------------------ | ---------------|
| Tensor-based bigram| 2.4343         |
| Neural net bigram  | 2.4906         |
| Underfitted MLP    | 2.1775         |
| Overfitted MLP     | 2.1483         |

Sample outputs:

**Tensor-based:**
```
cexze.
momasurailezitynn.
konimittain.
llayn.
ka.
da.
staiyaubrtthrigotai.
moliellavo.
ke.
teda.
ka.
emimmsade.
enkaviyny.
ftlspihinivenvorhlasu.
dsor.
br.
jol.
pen.
aisan.
ja.
```

**Neural net:**
```
min.
ela.
h.
sthay.
sxinienge.
made.
aleonnomin.
besaikarien.
yanjazien.
t.
lafannn.
cavin.
falis.
yoorlesiyndassemimah.
mizelxy.
mis.
ten.
odan.
h.
h.
```   

**Underfitted MLP:**
```
carpazfan.
jhavith.
miliatalykslanden.
jazhube.
delyah.
jareei.
ner.
keahciaiiv.
kaleigh.
ham.
joce.
quintis.
lilah.
jadia.
wavero.
dearynix.
kaeslinsley.
dae.
iia.
gian.
```

**Overfitted MLP:**
```
carmah.
amelle.
khyirlhi.
tate.
salaysa.
jazon.
nadeliah.
jareei.
ner.
kentzerian.
kaleigh.
ham.
evin.
quinn.
salin.
alvin.
quinte.
madearysia.
kael.
dusti.
```

## What I learned

- Implemented bigrams in two different ways (using tensors and using neural networks)
- Optimized the neural network by minimizing the loss function
- Concluded that regardless the method of implementation, bigrams are not efficient for creating character-level language models
- Implemented MLPs with different dimensions (one underfitting the data and the other overfitting it)
- Compared the difference between the loss of the bigram-based model to MLP-based models
- Understood the scalabality of MLPs by changing their dimensions

## Credits

Based on Andrej Karpathy's makemore: 
https://github.com/karpathy/makemore
