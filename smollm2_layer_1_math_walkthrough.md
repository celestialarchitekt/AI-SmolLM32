# SmolLM2-135M Layer 1 Math Walkthrough: `"I am happy"`

## Architecture Setup (Layer 1)

* **Hidden size ($d_{\text{model}}$):** `576`
* **Attention type:** Grouped-Query Attention (GQA) with **9 Query heads** ($q_1 \dots q_9$), **3 Key/Value heads** ($k_0 \dots k_2$ and $v_0 \dots v_2$), and **head dimension** `64` ($\sqrt{64} = 8$).
* **Positional encoding:** RoPE applied to $q$ and $k$ vectors (not $v$).
* **GQA Head Grouping:**
  * **Group 1 ($q_1, q_2, q_3$)** uses $k_0, v_0$
  * **Group 2 ($q_4, q_5, q_6$)** uses $k_1, v_1$
  * **Group 3 ($q_7, q_8, q_9$)** uses $k_2, v_2$

Each token enters Layer 1 as a `576`-dim vector $x_{\text{in}}^t$, passes through `RMSNorm`, and projects into nine `64`-dim Query vectors ($q_1^t \dots q_9^t$), three `64`-dim Key vectors ($k_0^t, k_1^t, k_2^t$), and three `64`-dim Value vectors ($v_0^t, v_1^t, v_2^t$).

---

## 1. Token `"I"` (Position 0 — Sees Only `"I"`)

### Scaled Dot-Product Scores (1 unmasked key per head)

* **Group 1 ($h = 1, 2, 3$):**

$$
s_{h,\text{I}} = \frac{q_h^{\text{I}} \cdot k_0^{\text{I}}}{8}
$$

* **Group 2 ($h = 4, 5, 6$):**

$$
s_{h,\text{I}} = \frac{q_h^{\text{I}} \cdot k_1^{\text{I}}}{8}
$$

* **Group 3 ($h = 7, 8, 9$):**

$$
s_{h,\text{I}} = \frac{q_h^{\text{I}} \cdot k_2^{\text{I}}}{8}
$$

### Softmax Weights (per head $h = 1 \dots 9$)

Positions 1 (`"am"`) and 2 (`"happy"`) are masked with $-\infty$, leaving one valid score:

$$
w_{h,\text{I}} = 1.0
$$

### Head Outputs ($\text{out}_1 \dots \text{out}_9$, each `64` dims)

* **Group 1 ($h = 1, 2, 3$):** $\text{out}_h^{\text{I}} = 1.0 \cdot v_0^{\text{I}}$
* **Group 2 ($h = 4, 5, 6$):** $\text{out}_h^{\text{I}} = 1.0 \cdot v_1^{\text{I}}$
* **Group 3 ($h = 7, 8, 9$):** $\text{out}_h^{\text{I}} = 1.0 \cdot v_2^{\text{I}}$

---

## 2. Token `"am"` (Position 1 — Sees `"I"` and `"am"`)

### Scaled Dot-Product Scores (2 unmasked keys per head)

* **Group 1 ($h = 1, 2, 3$):**

$$
s_{h,\text{I}} = \frac{q_h^{\text{am}} \cdot k_0^{\text{I}}}{8}, \quad s_{h,\text{am}} = \frac{q_h^{\text{am}} \cdot k_0^{\text{am}}}{8}
$$

* **Group 2 ($h = 4, 5, 6$):**

$$
s_{h,\text{I}} = \frac{q_h^{\text{am}} \cdot k_1^{\text{I}}}{8}, \quad s_{h,\text{am}} = \frac{q_h^{\text{am}} \cdot k_1^{\text{am}}}{8}
$$

* **Group 3 ($h = 7, 8, 9$):**

$$
s_{h,\text{I}} = \frac{q_h^{\text{am}} \cdot k_2^{\text{I}}}{8}, \quad s_{h,\text{am}} = \frac{q_h^{\text{am}} \cdot k_2^{\text{am}}}{8}
$$

### Softmax Weights (2 weights summing to `1.0` per head $h = 1 \dots 9$)

Position 2 (`"happy"`) is masked with $-\infty$:

$$
[w_{h,\text{I}},\; w_{h,\text{am}}] = \text{softmax}([s_{h,\text{I}},\; s_{h,\text{am}}])
$$

### Head Outputs ($\text{out}_1 \dots \text{out}_9$, each `64` dims)

* **Group 1 ($h = 1, 2, 3$):** $\text{out}_h^{\text{am}} = w_{h,\text{I}} \, v_0^{\text{I}} + w_{h,\text{am}} \, v_0^{\text{am}}$
* **Group 2 ($h = 4, 5, 6$):** $\text{out}_h^{\text{am}} = w_{h,\text{I}} \, v_1^{\text{I}} + w_{h,\text{am}} \, v_1^{\text{am}}$
* **Group 3 ($h = 7, 8, 9$):** $\text{out}_h^{\text{am}} = w_{h,\text{I}} \, v_2^{\text{I}} + w_{h,\text{am}} \, v_2^{\text{am}}$

---

## 3. Token `"happy"` (Position 2 — Sees `"I"`, `"am"`, and `"happy"`)

### Scaled Dot-Product Scores (3 unmasked keys per head)

* **Group 1 ($h = 1, 2, 3$):**

$$
s_{h,\text{I}} = \frac{q_h^{\text{happy}} \cdot k_0^{\text{I}}}{8}, \quad s_{h,\text{am}} = \frac{q_h^{\text{happy}} \cdot k_0^{\text{am}}}{8}, \quad s_{h,\text{happy}} = \frac{q_h^{\text{happy}} \cdot k_0^{\text{happy}}}{8}
$$

* **Group 2 ($h = 4, 5, 6$):**

$$
s_{h,\text{I}} = \frac{q_h^{\text{happy}} \cdot k_1^{\text{I}}}{8}, \quad s_{h,\text{am}} = \frac{q_h^{\text{happy}} \cdot k_1^{\text{am}}}{8}, \quad s_{h,\text{happy}} = \frac{q_h^{\text{happy}} \cdot k_1^{\text{happy}}}{8}
$$

* **Group 3 ($h = 7, 8, 9$):**

$$
s_{h,\text{I}} = \frac{q_h^{\text{happy}} \cdot k_2^{\text{I}}}{8}, \quad s_{h,\text{am}} = \frac{q_h^{\text{happy}} \cdot k_2^{\text{am}}}{8}, \quad s_{h,\text{happy}} = \frac{q_h^{\text{happy}} \cdot k_2^{\text{happy}}}{8}
$$

### Softmax Weights (3 weights summing to `1.0` per head $h = 1 \dots 9$)

$$
[w_{h,\text{I}},\; w_{h,\text{am}},\; w_{h,\text{happy}}] = \text{softmax}([s_{h,\text{I}},\; s_{h,\text{am}},\; s_{h,\text{happy}}])
$$

### Head Outputs ($\text{out}_1 \dots \text{out}_9$, each `64` dims)

* **Group 1 ($h = 1, 2, 3$):** $\text{out}_h^{\text{happy}} = w_{h,\text{I}} \, v_0^{\text{I}} + w_{h,\text{am}} \, v_0^{\text{am}} + w_{h,\text{happy}} \, v_0^{\text{happy}}$
* **Group 2 ($h = 4, 5, 6$):** $\text{out}_h^{\text{happy}} = w_{h,\text{I}} \, v_1^{\text{I}} + w_{h,\text{am}} \, v_1^{\text{am}} + w_{h,\text{happy}} \, v_1^{\text{happy}}$
* **Group 3 ($h = 7, 8, 9$):** $\text{out}_h^{\text{happy}} = w_{h,\text{I}} \, v_2^{\text{I}} + w_{h,\text{am}} \, v_2^{\text{am}} + w_{h,\text{happy}} \, v_2^{\text{happy}}$

---

## 4. Finishing Layer 1 (Applied Independently to Each Token $t \in \{\text{"I"}, \text{"am"}, \text{"happy"}\}$)

### Step 1: Concatenate 9 Heads (`9 × 64 = 576`) & Output Projection (`576 -> 576`)

$$
\text{attn\_out}^t = [\text{out}_1^t,\; \text{out}_2^t,\; \dots,\; \text{out}_9^t] \, W_O \in \mathbb{R}^{576}
$$

### Step 2: Attention Residual Add

$$
x_{\text{mid}}^t = x_{\text{in}}^t + \text{attn\_out}^t \in \mathbb{R}^{576}
$$

### Step 3: Post-Attention RMSNorm & SwiGLU MLP (`576 -> 1536 -> 576`, no bias)

$$
u^t = \text{RMSNorm}(x_{\text{mid}}^t) \in \mathbb{R}^{576}
$$

$$
\text{mlp\_out}^t = \big(\text{SiLU}(u^t W_{\text{gate}}) \odot (u^t W_{\text{up}})\big) W_{\text{down}} \in \mathbb{R}^{576}
$$

### Step 4: MLP Residual Add (Final Layer 1 Output $\rightarrow$ Input to Layer 2)

$$
x_{\text{out}}^t = x_{\text{mid}}^t + \text{mlp\_out}^t \in \mathbb{R}^{576}
$$