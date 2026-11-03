# MCA III Semester – Lab Sheet 08: Solved Mini Projects

This package covers both projects listed in the supplied lab sheet:

1. **Handwritten Digit Recognition using Neural Networks (MNIST)**
2. **Q-Learning Based Maze Solver**

## Included
- Readable Python source code for both projects.
- Training/evaluation routines, learning-curve plotting, model saving, and result comparison.
- A trained Q-table for the maze project, plus its training plot and sample route.
- A report with methodology, observations, limitations, and viva questions.

## Important note about the MNIST model
The MNIST training script is complete and saves a trained Keras model as `mnist_digit_model.keras` after execution. The current execution environment does not have TensorFlow installed and has no MNIST dataset cache, so a genuinely trained MNIST model is **not** falsely included. Install the requirements and run the script once; it downloads MNIST through Keras, trains, evaluates, plots curves, and saves the model.

## Run
Python 3.11+ recommended.

```bash
pip install -r requirements.txt
python project1_mnist/mnist_dnn.py
python project2_q_learning/maze_q_learning.py
```

The MNIST script uses TensorFlow/Keras. The maze project uses NumPy and Matplotlib only.

## Files
- `project1_mnist/mnist_dnn.py` – MNIST DNN source.
- `project2_q_learning/maze_q_learning.py` – Q-learning maze source.
- `results/` – generated maze results and learning curve.
- `documentation/` – project report and viva preparation.
