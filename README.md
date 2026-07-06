# makemore

A Character-level language model built while working through Andrej Karpathy's "Neural Networks: Zero to Hero" series. Implements the makemore models. As of now, the character-level language model is implemented using bigram counting models only. This model was trained on a dataset of names to generate new, name-like words.


## Status

Currently bigram-only. Next: MLP-based character model (in progress).


## What's in here

- `bigram/bigram_using_tensors.ipynb` — count-based bigram model, implemented using tensor functions available in the pytorch library.
- `bigram/bigram_using_neural_nets.ipynb` — count-based bigram model, implemented using a simple single-layered neural network.
- `bigram/plot_tensor.ipynb` — plots the bigram model in the form of a (27,27) tensor
- `names.txt` — training data (32K names)
- `bigram/tensor.png` — the image of the tensor created in `plot_tensor.ipynb`


## How it works

In the bigram implementation, the model learns the probability of the next character given the previous one, by counting co-occurences directly from the training data / learning them via gradient descent. Sampling from this probability distribution generates new names character by character until an end token is produced.


## Usage

```bash
git clone https://github.com/lalitadityamalik/makemore.git
cd makemore/bigram
jupyter notebook bigram_using_tensors.ipynb
jupyter notebook bigram_using_neural_nets.ipynb
jupyter notebook plot_tensor.ipynb
```


## Results

| Model              | Avg NLL (loss) |
| ------------------ | ---------------|
| Tensor-based bigram| 2.4343         |
| Neural net bigram  | 2.4906         |

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

## What I learned

- Implemented bigrams in two different ways (using tensors and using neural networks)
- Optimized the neural network by reducing loss function
- Concluded that no matter the method of implementation, bigrams are not efficient for creating character-level language models

## Credits

Based on Andrej Karpathy's makemore: 
https://github.com/karpathy/makemore
