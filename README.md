# gpt.c

gpt implemented in C [ongoing dev]

- [X] Implement matrix operations
- [X] Build a basic feed-forward neural network  
- [X] Develop backpropagation 
- [X] Gradient descent
- [X] Implement ReLU and Softmax
- [X] Loss function MSE
- [X] XOR Test
- [X] Add memory management ~~(object tracking, cleanup)~~ (slot system, objects occupy limited slots)
- [X] Construct forward and backward pass logic
- [X] MNIST Test
- [X] Implement Batching (major speedups)
- [X] Implemented GELU, Leaky RELU (all done as part of testing)
- [X] Implement iterative stack based backward pass (didn't do much benefit/ so removed)
- [X] Test the MLP with character prediction (Issues encounters: network stabiliy)
- [X] Tinystories Test
- [X] Implement n dimensional tensors
- [X] Implement Self-Attention Mechanism
- [ ] Build Positional Encoding  
- [ ] Develop Multi-Head Attention  
- [ ] Implement Layer Normalization  
- [ ] Stack Transformer Blocks  
- [ ] Build a tokenization system (BPE)    
- [ ] Implement text generation (Greedy Decoding)  
- [ ] Add Beam Search for better inference  
- [ ] Add performance metrics (perplexity, loss tracking)
- [ ] Implement data batching and shuffling

Current State:

- ~~too much object reallocation, design needs to change~~
- ~~Gradients are not converging properly~~
- ~~MNIST Test failed because of memory leaks.~~
- ~~Slow network convergence for large MLP~~
- ~~Network facing vanishing gradient issue~~
