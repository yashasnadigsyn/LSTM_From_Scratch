-  As discussed in the previous chapters, We can solve the exploding gradients problem with gradient clipping, but we still suffer from vanishing gradients problem. LSTMs (Long Short-Term Memory networks) were designed specifically to address this problem.

## The Memory Cell
- LSTMs are essentially RNNs with a special "memory cell" that replaces the ordinary recurrent node in standard RNNs.
- This memory cell is the heart of the LSTM and provides a mechanism for selectively remembering or forgetting information over time.
- Each LSTM contains both long-term memory (weights) and short-term memory (activations), as well as a special medium-term memory.
- The memory cell has an internal state and a number of "gates" that control the flow of information into and out of the cell.
  - **Input Gate:** How much of the new input should affect the internal state.
  - **Forget Gate:** Whether the current value of the memory should be flushed (set to zero).
  - **Output Gate:** Whether the internal state of the neuron is allowed to affect the cell's output.

- The purpose of the "gates" is to selectively modify the hidden state and thus enable longer term learning,  and help with the vanishing gradient problem.

## Input Gate, Forget Gate and Output Gate
- The data feeding into the LSTM gates are the input at the current time step and the hidden state of the previous time step.
	![[LSTM1.png]]
- It = σ(Xt Wxi + Ht-1 Whi + bi)
- Ft = σ(Xt Wxf + Ht-1 Whf + bf)
- Ot = σ(Xt Wxo + Ht-1 Who + bo)

- It, Ft, Ot ∈ R<sup>(n x h)</sup>: The values of the input, forget, and output gates at time step t. The numbers are between 0 and 1 due to the sigmoid function.
- σ: The sigmoid activation function. This squashes the values between 0 and 1, representing the degree to which the gate is "open" or "closed."
- Xt ∈ R<sup>(n x d)</sup>: The input at time step t. (Batch size n, input features d)
- Ht-1 ∈ R<sup>(n x h)</sup>: The hidden state from the previous time step. (Batch size n, hidden units h)
- Wxi, Wxf, Wxo ∈ R<sup>(d x h)</sup>: Weight matrices connecting the input Xt to the input, forget, and output gates, respectively.
- Whi, Whf, Who ∈ R<sup>(h x h)</sup>: Weight matrices connecting the previous hidden state Ht-1 to the input, forget, and output gates, respectively.
- bi, bf, bo ∈ R<sup>(1 x h)</sup>: Bias vectors for the input, forget, and output gates, respectively.

## Input Node
- Ct = tanh(Xt Wxc + Ht-1 Whc + bc)
- The input node is similar to the hidden state in a standard RNN, but it's not directly used as the hidden state. It's an intermediate value that is modulated by the input gate.
![[LSTM2.png]]
## Memory Cell Internal State
- In LSTMs, the input gate I𝑡 governs how much we take new data into account via ˜C𝑡 and
the forget gate F𝑡 addresses how much of the old cell internal state C𝑡 −1 ∈ R𝑛×ℎ we retain.
- Ct = Ft ⊙ Ct-1 + It ⊙ Ct
- If the forget gate is always 1 and the input gate is always 0, the memory cell internal state C𝑡 −1 will remain constant forever, passing unchanged to each subsequent time step. However, input gates and forget gates give the model the flexibility of being able to learn when to keep this value unchanged and when to perturb it in response to subsequent inputs. 
- In practice, this design alleviates the vanishing gradient problem, resulting in models that are much easier to train, especially when facing datasets with long sequence lengths.
- We arrive at this flow diagram
![[LSTM3.png]]
## Hidden State
- Ht = Ot ⊙ tanh(Ct)
  ![[LSTM4.png]]
