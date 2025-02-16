\text{ReLU}(x) = \begin{cases}
    0, & \text{if } x < 0 \\
    x, & \text{if } x \geq 0
\end{cases}

\text{Leaky ReLU}(x) = \begin{cases}
    \alpha x, & \text{if } x < 0 \\
    x, & \text{if } x \geq 0
\end{cases}

- Where α is a small constant, typically around 0.01. 
- We can also represent ReLU using a simpler expression using the max function. Leaky ReLU is also expressed the same.

\text{ReLU}(x) = \max(0, x)

\text{Leaky ReLU}(x) = \max(\alpha x, x)
