# 🌉 Euler–Bernoulli Beam Solver Using Physics-Informed Neural Networks (PINNs)

A Physics-Informed Neural Network (PINN) implementation in **PyTorch** for solving the **Euler–Bernoulli beam equation** under a uniformly distributed load (UDL). Unlike traditional numerical methods such as the Finite Element Method (FEM), this model learns the beam deflection by embedding the governing differential equation and boundary conditions directly into the neural network loss function using automatic differentiation.

---

## 📌 Overview

Physics-Informed Neural Networks (PINNs) combine deep learning with the underlying laws of physics by incorporating differential equations into the loss function. This allows the neural network to approximate the solution without requiring labeled simulation or experimental data.

This project solves the **Euler–Bernoulli beam equation**

\[
EI\frac{d^4y}{dx^4}=q
\]

for a **simply supported beam** subjected to a **uniformly distributed load**.

---

## 🚀 Features

- Physics-Informed Neural Network implemented using **PyTorch**
- Automatic differentiation for first- to fourth-order derivatives
- Mesh-free solution without FEM or finite difference methods
- Physics residual and boundary condition loss formulation
- Random collocation point sampling during training
- Visualization of:
  - Training Loss
  - Beam Deflection
  - Bending Moment Diagram

---

## 🛠 Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib

---

## 🧠 Neural Network Architecture

| Component | Configuration |
|----------|---------------|
| Input Layer | 1 neuron |
| Hidden Layers | 2 |
| Hidden Units | 64 neurons each |
| Activation | Tanh |
| Output Layer | 1 neuron |

Training Details:

- Optimizer: Adam
- Learning Rate: `1e-5`
- Epochs: `10,000`
- Collocation Points: `100` sampled randomly every epoch

---

## 📖 Methodology

The network is trained by minimizing the total loss

\[
\mathcal{L}=\mathcal{L}_{physics}+1000\times\mathcal{L}_{boundary}
\]

### Physics Loss

The governing equation

\[
\frac{d^4y}{dx^4}-\frac{q}{EI}=0
\]

is enforced by computing the fourth derivative using automatic differentiation.

### Boundary Loss

For a simply supported beam,

- \(y(0)=0\)
- \(y(L)=0\)
- \(y''(0)=0\)
- \(y''(L)=0\)

These constraints are incorporated directly into the loss function.

---

## 📊 Results

The trained PINN successfully learns the beam deflection while satisfying both the governing differential equation and the boundary conditions.

The notebook generates:

- 📉 Loss vs Epochs
- 📈 Beam Deflection Curve
- 📈 Bending Moment Diagram

The predicted beam deflection closely matches the analytical solution, achieving a **relative L2 error of approximately 0.77%**.

---

## 📂 Project Structure

```
.
├── PINN2.ipynb
├── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Install the required libraries

```bash
pip install torch numpy matplotlib
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open **PINN2.ipynb** and run all cells.

---

## 🔬 Future Improvements

- Variable distributed loads
- Cantilever and fixed beam boundary conditions
- Beam vibration analysis
- Adaptive collocation sampling
- Comparison with FEM solvers
- GPU acceleration

---

## 📚 References

- M. Raissi, P. Perdikaris, G. Karniadakis, **Physics-Informed Neural Networks**, Journal of Computational Physics, 2019.
- Euler–Bernoulli Beam Theory

---

## 👨‍💻 Author

**Abhijay Kashyap**

B.E. Civil Engineering  
BITS Pilani Hyderabad Campus

GitHub: *Add your GitHub profile link here*

---

⭐ If you found this project useful, consider starring the repository.
