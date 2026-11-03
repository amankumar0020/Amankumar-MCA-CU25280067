# Lab Sheet 08 – Mini Projects on Deep Learning and Reinforcement Learning

**Programme:** MCA III Semester (Session 2026–2027)

## Objective
- Understand neural networks and reinforcement-learning concepts.
- Build intelligent systems using deep-learning frameworks.
- Compare traditional machine learning with deep learning.

## Project 1: Handwritten Digit Recognition using Neural Networks

### Aim
Develop a Deep Neural Network (DNN) using the MNIST dataset to recognize handwritten digits and evaluate model accuracy.

### Dataset and preprocessing
MNIST contains grayscale images of handwritten digits (0–9), each of size 28 × 28 pixels. The Keras dataset loader supplies 60,000 training images and 10,000 test images. Pixel values are converted from the 0–255 range to floating-point values between 0 and 1.

### Model
The implementation uses a feed-forward DNN:
1. Flatten: 28 × 28 image to 784 features.
2. Dense layer: 256 neurons, ReLU activation.
3. Dropout: 0.20.
4. Dense layer: 128 neurons, ReLU activation.
5. Dropout: 0.20.
6. Output: 10 neurons, softmax activation.

Adam optimizer and sparse categorical cross-entropy loss are used. Training uses 10 epochs, batch size 128, and a 10% validation split.

### Evaluation
The script evaluates loss and accuracy on the held-out MNIST test set, prints sample predictions, plots training/validation accuracy and loss, and saves the trained model to `trained_models/mnist_digit_model.keras`.

**Execution note:** TensorFlow is not installed in the environment used to prepare this archive, so the MNIST network has not been trained here and no fabricated test accuracy or model file is claimed. Run the supplied script after installing `requirements.txt` to download MNIST, train, evaluate, and create the model and plots.

## Project 2: Q-Learning Based Maze Solver

### Aim
Develop a reinforcement-learning agent that learns a path from the start cell to the goal cell using Q-learning and reward optimization.

### Environment
A 5 × 5 grid is used. `0` represents an open cell and `1` represents a wall. Start is (0, 0) and goal is (4, 4). The four actions are UP, DOWN, LEFT, and RIGHT.

### Reward design
- Reaching the goal: +100
- Valid non-goal move: −1
- Hitting a wall or moving outside the maze: −5

### Q-learning update
`Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]`

Learning rate α = 0.15, discount factor γ = 0.95. Epsilon-greedy exploration starts at 1.0, decays by 0.997 per episode, and has a minimum of 0.03. The agent is trained for 5,000 episodes.

### Result
The included Q-table is trained from the specified maze and seed. The extracted greedy route is:
`(0, 0) -> (0, 1) -> (0, 2) -> (1, 2) -> (2, 2) -> (2, 3) -> (2, 4) -> (3, 4) -> (4, 4)`

Reached goal: **True**  
Number of moves: **8**

The training curve and route visualization are included in `results/`.

## Comparison: Traditional ML and Deep Learning

| Aspect | Traditional ML / Tabular Q-learning | Deep Neural Network |
|---|---|---|
| Representation | Often uses manually designed features or a table of state-action values | Learns layered representations from input data |
| Input in these projects | Discrete maze states and actions | 28 × 28 pixel images |
| Learning objective | Maximize cumulative reward | Minimize classification loss |
| Output | Preferred action / route | Digit class from 0 to 9 |
| Main strength | Simple and interpretable for small state spaces | Handles high-dimensional image inputs |
| Limitation | Q-table grows with the number of states and actions | Needs more computation and labelled examples |

## Observations
1. Normalizing image pixels makes neural-network optimization more manageable.
2. ReLU provides non-linearity in the hidden layers; softmax produces class probabilities.
3. Dropout is used to reduce overfitting.
4. In Q-learning, exploration is important early in training; epsilon decay gradually shifts behavior toward exploitation.
5. The reward design influences the route learned by the agent.
6. A tabular Q-table is suitable for this small maze but is not practical for very large or continuous state spaces.

## Conclusion
Both mini projects demonstrate different forms of intelligent learning. The MNIST DNN performs supervised image classification, while the maze agent learns a policy through trial-and-error interaction and rewards. The scripts save models and plots so their results can be evaluated and documented.

## Viva Questions and Answers

1. **What is deep learning?**  
   Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn patterns from data.

2. **What is MNIST?**  
   MNIST is a dataset of handwritten digit images from 0 to 9, commonly used for image-classification experiments.

3. **Why normalize image pixels?**  
   Normalization scales input values to a smaller range, which generally helps stable and efficient training.

4. **What is an activation function?**  
   It introduces non-linearity so a neural network can learn complex relationships. This project uses ReLU and softmax.

5. **What is dropout?**  
   Dropout randomly disables a fraction of units during training to help reduce overfitting.

6. **What is reinforcement learning?**  
   It is a learning approach in which an agent interacts with an environment and learns from rewards and penalties.

7. **What is Q-learning?**  
   Q-learning is a model-free, off-policy reinforcement-learning algorithm that estimates the value of taking an action in a state.

8. **What is the Q-table?**  
   It stores estimated action values for each state-action pair.

9. **What is epsilon-greedy selection?**  
   It chooses a random action with probability epsilon and the best-known action otherwise.

10. **What do alpha and gamma mean?**  
    Alpha is the learning rate; gamma is the discount factor for future rewards.

11. **Why is a penalty given for hitting a wall?**  
    It discourages invalid actions and helps the agent learn legal paths.

12. **What is the difference between supervised and reinforcement learning?**  
    Supervised learning uses labelled examples; reinforcement learning learns by interacting with an environment and receiving rewards.

13. **Why plot learning curves?**  
    They help inspect how model performance or reward changes during training.

14. **Why save a trained model?**  
    Saving allows the learned parameters to be reused without training from scratch.

15. **What is overfitting?**  
    Overfitting occurs when a model performs well on training data but generalizes poorly to unseen data.
