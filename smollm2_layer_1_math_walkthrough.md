# SmolLM2-135M Layer 1 Math Walkthrough: `"I am happy"`

## Architecture Setup (Layer 1)

* **Hidden size (`d_model`):** `576`
* **Attention type:** Grouped-Query Attention (GQA) with **9 Query heads** (`q1` ... `q9`), **3 Key/Value heads** (`k0` ... `k2` and `v0` ... `v2`), and **head dimension** `64` (`sqrt(64) = 8`).
* **Positional encoding:** RoPE applied to `q` and `k` vectors (not `v`).
* **GQA Head Grouping:**
  * **Group 1 (`q1, q2, q3`)** uses `k0, v0`
  * **Group 2 (`q4, q5, q6`)** uses `k1, v1`
  * **Group 3 (`q7, q8, q9`)** uses `k2, v2`

Each token enters Layer 1 as a `576`-dim vector `x_in[t]`, passes through `RMSNorm`, and projects into nine `64`-dim Query vectors (`q1` ... `q9`), three `64`-dim Key vectors (`k0, k1, k2`), and three `64`-dim Value vectors (`v0, v1, v2`).

---

## 1. Token `"I"` (Position 0 — Sees Only `"I"`)

### Scaled Dot-Product Scores (1 unmasked key per head)
* **Group 1 (`h = 1, 2, 3`):**
  ```text
  s[h, I] = (q[h, I] · k[0, I]) / 8
  ```
* **Group 2 (`h = 4, 5, 6`):**
  ```text
  s[h, I] = (q[h, I] · k[1, I]) / 8
  ```
* **Group 3 (`h = 7, 8, 9`):**
  ```text
  s[h, I] = (q[h, I] · k[2, I]) / 8
  ```

### Softmax Weights (per head `h = 1 ... 9`)
Positions 1 (`"am"`) and 2 (`"happy"`) are masked with `-inf`, leaving one valid score:
```text
w[h, I] = 1.0
```

### Head Outputs (`out1` ... `out9`, each `64` dims)
* **Group 1 (`h = 1, 2, 3`):** `out[h, I] = 1.0 * v[0, I]`
* **Group 2 (`h = 4, 5, 6`):** `out[h, I] = 1.0 * v[1, I]`
* **Group 3 (`h = 7, 8, 9`):** `out[h, I] = 1.0 * v[2, I]`

---

## 2. Token `"am"` (Position 1 — Sees `"I"` and `"am"`)

### Scaled Dot-Product Scores (2 unmasked keys per head)
* **Group 1 (`h = 1, 2, 3`):**
  ```text
  s[h, I]  = (q[h, am] · k[0, I])  / 8
  s[h, am] = (q[h, am] · k[0, am]) / 8
  ```
* **Group 2 (`h = 4, 5, 6`):**
  ```text
  s[h, I]  = (q[h, am] · k[1, I])  / 8
  s[h, am] = (q[h, am] · k[1, am]) / 8
  ```
* **Group 3 (`h = 7, 8, 9`):**
  ```text
  s[h, I]  = (q[h, am] · k[2, I])  / 8
  s[h, am] = (q[h, am] · k[2, am]) / 8
  ```

### Softmax Weights (2 weights summing to `1.0` per head `h = 1 ... 9`)
Position 2 (`"happy"`) is masked with `-inf`:
```text
[w[h, I], w[h, am]] = softmax([s[h, I], s[h, am]])
```

### Head Outputs (`out1` ... `out9`, each `64` dims)
* **Group 1 (`h = 1, 2, 3`):** `out[h, am] = w[h, I] * v[0, I] + w[h, am] * v[0, am]`
* **Group 2 (`h = 4, 5, 6`):** `out[h, am] = w[h, I] * v[1, I] + w[h, am] * v[1, am]`
* **Group 3 (`h = 7, 8, 9`):** `out[h, am] = w[h, I] * v[2, I] + w[h, am] * v[2, am]`

---

## 3. Token `"happy"` (Position 2 — Sees `"I"`, `"am"`, and `"happy"`)

### Scaled Dot-Product Scores (3 unmasked keys per head)
* **Group 1 (`h = 1, 2, 3`):**
  ```text
  s[h, I]     = (q[h, happy] · k[0, I])     / 8
  s[h, am]    = (q[h, happy] · k[0, am])    / 8
  s[h, happy] = (q[h, happy] · k[0, happy]) / 8
  ```
* **Group 2 (`h = 4, 5, 6`):**
  ```text
  s[h, I]     = (q[h, happy] · k[1, I])     / 8
  s[h, am]    = (q[h, happy] · k[1, am])    / 8
  s[h, happy] = (q[h, happy] · k[1, happy]) / 8
  ```
* **Group 3 (`h = 7, 8, 9`):**
  ```text
  s[h, I]     = (q[h, happy] · k[2, I])     / 8
  s[h, am]    = (q[h, happy] · k[2, am])    / 8
  s[h, happy] = (q[h, happy] · k[2, happy]) / 8
  ```

### Softmax Weights (3 weights summing to `1.0` per head `h = 1 ... 9`)
```text
[w[h, I], w[h, am], w[h, happy]] = softmax([s[h, I], s[h, am], s[h, happy]])
```

### Head Outputs (`out1` ... `out9`, each `64` dims)
* **Group 1 (`h = 1, 2, 3`):** `out[h, happy] = w[h, I] * v[0, I] + w[h, am] * v[0, am] + w[h, happy] * v[0, happy]`
* **Group 2 (`h = 4, 5, 6`):** `out[h, happy] = w[h, I] * v[1, I] + w[h, am] * v[1, am] + w[h, happy] * v[1, happy]`
* **Group 3 (`h = 7, 8, 9`):** `out[h, happy] = w[h, I] * v[2, I] + w[h, am] * v[2, am] + w[h, happy] * v[2, happy]`

---

## 4. Finishing Layer 1 (Applied Independently to Each Token `t` in `{"I", "am", "happy"}`)

1. **Concatenate 9 Heads (`9 × 64 = 576`) & Output Projection (`576 -> 576`):**
   ```text
   attn_out[t] = concat(out[1, t], out[2, t], ..., out[9, t]) @ W_O   # shape: (576,)
   ```

2. **Attention Residual Add:**
   ```text
   x_mid[t] = x_in[t] + attn_out[t]                                   # shape: (576,)
   ```

3. **Post-Attention RMSNorm & SwiGLU MLP (`576 -> 1536 -> 576`, no bias):**
   ```text
   u[t]       = RMSNorm(x_mid[t])                                     # shape: (576,)
   mlp_out[t] = (SiLU(u[t] @ W_gate) * (u[t] @ W_up)) @ W_down        # shape: (576,)
   ```

4. **MLP Residual Add (Final Layer 1 Output -> Input to Layer 2):**
   ```text
   x_out[t] = x_mid[t] + mlp_out[t]                                   # shape: (576,)
   ```