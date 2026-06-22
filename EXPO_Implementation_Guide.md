# EXPO: Expressive Policy Optimization — Complete Implementation Guide

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Components](#2-architecture-components)
3. [Core Algorithms](#3-core-algorithms)
4. [Network Architectures](#4-network-architectures)
5. [Training Pipeline](#5-training-pipeline)
6. [Loss Functions](#6-loss-functions)
7. [On-The-Fly Policy Extraction](#7-on-the-fly-policy-extraction)
8. [Replay Buffer & Data Management](#8-replay-buffer--data-management)
9. [Diffusion Policy (Base Policy)](#9-diffusion-policy-base-policy)
10. [Edit Policy](#10-edit-policy)
11. [Critic Network](#11-critic-network)
12. [Hyperparameters](#12-hyperparameters)
13. [Evaluation Protocol](#13-evaluation-protocol)
14. [Full Pseudocode](#14-full-pseudocode)
15. [Dependency Map](#15-dependency-map)

---

## 1. Overview

EXPO is an online RL algorithm for fine-tuning expressive policies (diffusion, flow-matching) that avoids direct backpropagation of value gradients through the long denoising chain. Instead, it constructs an **on-the-fly (OTF) policy** from two components:

- **Base Policy** (`π_base`): A large expressive policy (e.g., diffusion) trained with a stable imitation learning (IL) objective.
- **Edit Policy** (`π_edit`): A lightweight Gaussian policy that applies small perturbations to base actions to maximize Q-value.

The OTF policy samples N action candidates from the base policy, edits each with the edit policy, and selects the action with the highest Q-value for both environment interaction and TD backup.

### Key Insight

The expressive policy is **never** trained to maximize value directly. Value maximization is handled entirely by the edit policy and the argmax selection step, sidestepping gradient instability.

---

## 2. Architecture Components

### 2.1 Component Summary

| Component | Type | Role | Training Objective |
|-----------|------|------|--------------------|
| Base Policy `π_base(a\|s)` | Diffusion (DDPM) with residual blocks | Generate diverse, multi-modal actions | Imitation learning (denoising loss) |
| Edit Policy `π_edit(â\|s, a)` | Gaussian MLP | Refine base actions toward high Q-value | Entropy-regularized policy gradient (SAC-style) |
| Critic `Q_ϕ(s, a)` | Ensemble of MLPs | Estimate state-action values | TD learning with OTF target |
| Target Critic `Q_ϕ'(s, a)` | EMA copy of critic | Stable TD targets | Polyak averaging |

### 2.2 Data Flow Diagram

```
State s
  │
  ├──► Base Policy π_base(·|s) ──► N base actions {a_1, ..., a_N}
  │                                       │
  │                                       ▼
  ├──► Edit Policy π_edit(·|s, a_i) ──► N edits {â_1, ..., â_N}
  │                                       │
  │                                       ▼
  │                              ã_i = a_i + clip(â_i, -β, β)
  │                                       │
  │                                       ▼
  │                         Candidates = {a_1, ã_1, ..., a_N, ã_N}
  │                                       │
  └──► Critic Q_ϕ(s, ·) ────────────────► argmax Q_ϕ(s, candidate)
                                           │
                                           ▼
                                    Selected action a*
```

---

## 3. Core Algorithms

### 3.1 Algorithm 1: EXPO Main Loop

```
Input: Prior dataset D_data = {(s_i, a_i)}, optional pretrained π_base
Initialize: π_edit (random), Q_ϕ (random), Q_ϕ' (copy of Q_ϕ), UTD ratio G

while training:
    for each environment step t:
        # === COLLECT ROLLOUTS ===
        a* ← π_OTF(·|s_t, π_base, π_edit, ϕ')     # OTF action selection
        Execute a* in environment, observe r_t, s_{t+1}
        Store (s_t, a*, r_t, s_{t+1}) in replay buffer D

        # === UPDATE POLICY AND CRITIC ===
        for g = 1 to G:                              # G = UTD ratio
            Sample mini-batch (s, a, r, s') from D ∪ D_data
            
            # Critic update
            a*' ← π_OTF(·|s', π_base, π_edit, ϕ')   # OTF target action
            y = r + γ · Q_ϕ'(s', a*')                # TD target
            Update ϕ minimizing: L = (y - Q_ϕ(s, a))²

            # Target network update
            ϕ' ← τ · ϕ' + (1 - τ) · ϕ

        # Base policy update (last mini-batch)
        Update π_base with L_IL(π_base)               # Imitation learning

        # Edit policy update (last mini-batch)
        Sample â ~ π_edit(·|s, a), where a ~ π_base(·|s)
        Update π_edit maximizing: Q_ϕ(s, a + â) - α · log π_edit(â|s, a)
```

### 3.2 On-The-Fly Policy `π_OTF`

```
function π_OTF(·|s, π_base, π_edit, ϕ):
    for i = 1 to N:
        a_i ~ π_base(·|s)                           # Sample from base
        â_i ~ π_edit(·|s, a_i)                      # Sample edit
        â_i ← clip(â_i, -β, β)                      # Enforce edit distance
        ã_i = a_i + â_i                              # Apply edit
    
    candidates = {a_1, ã_1, a_2, ã_2, ..., a_N, ã_N}    # 2N candidates
    a* = argmax_{c ∈ candidates} Q_ϕ(s, c)
    return a*
```

---

## 4. Network Architectures

### 4.1 Base Policy (Diffusion — DDPM)

**Architecture**: Residual MLP with denoising network `ε_ψ`.

```
Input: noised action x_t, state s, timestep t
Output: predicted noise ε

Network structure:
  ├── State encoder: Linear(state_dim → 256)
  ├── Time embedding: sinusoidal embedding → Linear(embed_dim → 256)
  ├── Action input: Linear(action_dim → 256)
  │
  ├── Residual Block 1:
  │     ├── Linear(256 → 256) + LayerNorm + Mish
  │     ├── Linear(256 → 256) + LayerNorm + Mish
  │     └── Skip connection
  ├── Residual Block 2: (same structure)
  ├── Residual Block 3: (same structure)
  │
  └── Output: Linear(256 → action_dim)
```

**Config**:
- Hidden dim: 256
- Num residual blocks: 3
- Activation: Mish
- Normalization: LayerNorm
- Diffusion timesteps T: 10
- Beta schedule: Variance Preserving (VP)

### 4.2 Edit Policy (Gaussian MLP)

```
Input: state s, base action a (concatenated)
Output: mean μ and log_std of Gaussian distribution over â

Network structure:
  ├── Input: Linear(state_dim + action_dim → 256)
  ├── Hidden Layer 1: Linear(256 → 256) + ReLU
  ├── Hidden Layer 2: Linear(256 → 256) + ReLU
  ├── Hidden Layer 3: Linear(256 → 256) + ReLU
  │
  ├── Mean head: Linear(256 → action_dim)
  └── Log-std head: Linear(256 → action_dim)

Sampling: â = μ + σ · ε,  ε ~ N(0, I)
Then:     â = β · tanh(â)   # Scale to [-β, β]
```

**Config**:
- Hidden dim: 256
- Hidden layers: 3
- Activation: ReLU
- Dropout: 0.1 (Adroit only), None elsewhere
- Output: Squashed Gaussian (tanh) scaled by β

### 4.3 Critic Network (Q-Ensemble)

```
Input: state s, action a (concatenated)
Output: scalar Q-value

Single Q-network:
  ├── Linear(state_dim + action_dim → 256) + ReLU
  ├── Linear(256 → 256) + ReLU
  ├── Linear(256 → 256) + ReLU
  └── Linear(256 → 1)

Ensemble: 10 independent Q-networks
Min-Q: Take minimum over randomly sampled 2 of 10 (Num Min Q = 2)
```

**Config**:
- Ensemble size: 10
- Num Min Q (for pessimism): 2
- This follows the REDQ / Randomized Ensemble approach

---

## 5. Training Pipeline

### 5.1 Phase 1: Offline Pretraining (Optional)

Only the **base policy** is pretrained. No value function or edit policy pretraining.

```
for step = 1 to pretraining_steps:
    Sample (s, a) from D_data
    Compute diffusion loss L_IL on (s, a)
    Update π_base parameters ψ
```

**Pretraining steps by domain**:
| Domain | Steps |
|--------|-------|
| Robomimic | 200,000 |
| Adroit | 20,000 (or skip for EXPO online) |
| Antmaze | 500,000 |
| MimicGen | 200,000 |

### 5.2 Phase 2: Online Fine-Tuning

The full EXPO loop (Algorithm 1) with:
- Replay buffer initialized with offline data `D_data`
- Base policy initialized from pretraining (or random)
- Edit policy initialized randomly
- Critic initialized randomly

### 5.3 Update Schedule

Per environment step:
- **Critic**: Updated G times (UTD = 20)
- **Base policy**: Updated 1 time (on last mini-batch of G updates)
- **Edit policy**: Updated 1 time (on last mini-batch of G updates)
- **Target network**: Soft-updated G times (each critic step)

---

## 6. Loss Functions

### 6.1 Diffusion Imitation Learning Loss (Base Policy)

```
L_IL(ψ) = E_{t~U({1,...,T}), ε~N(0,I), (s,a)~D} [ ||ε - ε_ψ(√ᾱ_t · a + √(1-ᾱ_t) · ε, s, t)||² ]
```

Where:
- `t` is the diffusion timestep sampled uniformly from {1, ..., T}
- `ε` is Gaussian noise
- `ᾱ_t` is the cumulative product of the noise schedule `α_t = 1 - β_t`
- `ε_ψ` is the denoising network
- `D` is the combined replay buffer + offline data

**Important**: The base policy is trained on **all** data in the replay buffer (both offline and online collected), not just offline data. This allows the base policy to adapt to the improving behavior distribution.

### 6.2 Edit Policy Loss

```
L(π_edit) = -E_{(s,a)~D, â~π_edit(·|s,a)} [ Q_ϕ(s, a + â) - α · log π_edit(â|s, a) ]
```

Where:
- `a` is sampled from the base policy `π_base(·|s)` for the current state
- `â` is the edit sampled from `π_edit(·|s, a)`
- `α` is the entropy temperature (auto-tuned as in SAC)
- The edit is clipped: `â ← clip(â, -β, β)` before adding to `a`

**SAC-style entropy tuning**:
```
L(α) = -α · E[log π_edit(â|s, a) + target_entropy]
target_entropy = -dim(action_space)
```

### 6.3 Critic Loss (TD Learning)

```
L(ϕ) = E_{(s,a,r,s')~D} [ (r + γ · Q_ϕ'(s', a*') - Q_ϕ(s, a))² ]
```

Where:
- `a*' = π_OTF(·|s', π_base, π_edit, ϕ')` — the OTF action at the next state
- The target uses the **target critic** `Q_ϕ'`
- For the ensemble: randomly sample 2 of 10 Q-networks, take the minimum for the target

### 6.4 Gradient Flow Summary

```
Base Policy ψ:
    ∇ψ L_IL  (gradients from imitation loss only — NO Q-value gradients)

Edit Policy θ:
    ∇θ [Q_ϕ(s, a + â) - α log π_edit(â|s,a)]
    (gradients flow through Q into â into θ — single step, stable)

Critic ϕ:
    ∇ϕ (y - Q_ϕ(s, a))²
    (standard TD gradient, y is detached)
```

---

## 7. On-The-Fly Policy Extraction

### 7.1 Detailed Implementation

```python
def otf_action_selection(state, base_policy, edit_policy, critic, N, beta):
    """
    Select the value-maximizing action from base and edited candidates.
    
    Args:
        state: current state [batch_size, state_dim]
        base_policy: diffusion base policy
        edit_policy: Gaussian edit policy
        critic: Q-function ensemble
        N: number of action samples (default 8)
        beta: edit distance constraint
    
    Returns:
        best_action: [batch_size, action_dim]
    """
    batch_size = state.shape[0]
    
    # Repeat state for N samples: [batch_size * N, state_dim]
    state_repeated = state.unsqueeze(1).repeat(1, N, 1).reshape(-1, state.shape[-1])
    
    # Sample N actions from base policy
    base_actions = base_policy.sample(state_repeated)  # [batch_size * N, action_dim]
    
    # Get edits from edit policy
    edits = edit_policy.sample(state_repeated, base_actions)  # [batch_size * N, action_dim]
    edits = edits.clamp(-beta, beta)
    
    # Edited actions
    edited_actions = base_actions + edits  # [batch_size * N, action_dim]
    edited_actions = edited_actions.clamp(-1.0, 1.0)  # clip to action bounds
    
    # Concatenate base and edited: [batch_size * 2N, action_dim]
    all_actions = torch.cat([base_actions, edited_actions], dim=0)
    all_states = torch.cat([state_repeated, state_repeated], dim=0)
    
    # Evaluate Q-values
    q_values = critic.min_q(all_states, all_actions)  # [batch_size * 2N, 1]
    
    # Reshape: [batch_size, 2N]
    q_values = q_values.reshape(batch_size, 2 * N)
    all_actions = all_actions.reshape(batch_size, 2 * N, -1)
    
    # Select best action per batch element
    best_idx = q_values.argmax(dim=1)  # [batch_size]
    best_action = all_actions[torch.arange(batch_size), best_idx]
    
    return best_action
```

### 7.2 Usage Contexts

The OTF policy is used in **two** places:

1. **Environment rollout** (action selection): Use target critic `Q_ϕ'` for evaluation
2. **TD target computation**: For computing `a*'` at next state `s'` in the Bellman backup

Both must use the OTF extraction — using only the base policy for TD targets results in a SARSA-like update with significantly slower learning (see ablation in paper).

---

## 8. Replay Buffer & Data Management

### 8.1 Buffer Structure

```
Combined Buffer:
  ├── Offline Dataset D_data: {(s, a)} — fixed, loaded at start
  └── Online Replay Buffer D: {(s, a, r, s')} — grows during training

Sampling:
  - Sample uniformly from D ∪ D_data
  - Following RLPD: 50% from offline, 50% from online (configurable)
```

### 8.2 Transition Format

```python
transition = {
    'state': np.array,        # shape: (state_dim,)
    'action': np.array,       # shape: (action_dim,)
    'reward': float,          # scalar
    'next_state': np.array,   # shape: (state_dim,)
    'done': bool              # episode termination flag
}
```

### 8.3 Offline Data Handling

- Offline dataset may only contain `(s, a)` pairs (no rewards)
- For IL pretraining, only states and actions are needed
- For online training, offline data contributes to the IL loss for the base policy
- The offline data is retained in the buffer throughout training (but the paper shows EXPO can work even without retaining it — see ablation)

---

## 9. Diffusion Policy (Base Policy)

### 9.1 DDPM Forward Process

```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1 - ᾱ_t) · I)

Where:
  β_1, ..., β_T are the noise schedule (Variance Preserving)
  α_t = 1 - β_t
  ᾱ_t = Π_{i=1}^{t} α_i
```

### 9.2 DDPM Reverse Process (Sampling)

```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1, else z = 0
    x_{t-1} = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_ψ(x_t, s, t)) + σ_t · z

Where σ_t = √β_t (or √(β̃_t) for the variance-preserving schedule)
```

### 9.3 Training Objective

```
L_DDPM = E_{t, ε, (s,a)} [ ||ε - ε_ψ(√ᾱ_t · a + √(1-ᾱ_t) · ε, s, t)||² ]
```

### 9.4 Variance Preserving Schedule

```python
def vp_beta_schedule(T=10):
    beta_start = 0.0001
    beta_end = 0.02
    betas = np.linspace(beta_start, beta_end, T)
    return betas
```

### 9.5 Conditioning on State

The denoising network takes the state `s` as a conditioning input. The state is encoded and added/concatenated at every residual block or at the input layer.

---

## 10. Edit Policy

### 10.1 Gaussian Policy with Squashed Output

```python
class EditPolicy:
    def forward(self, state, base_action):
        x = concat(state, base_action)
        x = MLP(x)  # 3 hidden layers, 256 units each
        mean = mean_head(x)
        log_std = log_std_head(x).clamp(-20, 2)
        
        # Reparameterization
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        
        # Squash to [-β, β]
        edit = beta * torch.tanh(raw_action)
        
        # Log probability (with tanh correction)
        log_prob = normal.log_prob(raw_action) - torch.log(1 - (edit/beta)² + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return edit, log_prob
```

### 10.2 Edit Distance Constraint

The edit `â` is scaled to `[-β, β]` via the tanh squashing:
- **β = 0.05**: Small edits for tasks with good offline data (Robomimic, MimicGen, Antmaze online)
- **β = 0.1**: Moderate edits (Robomimic offline-to-online)
- **β = 0.7**: Large edits for high-exploration tasks (Adroit)
- **β = 0.0**: No edits (Antmaze offline-to-online — base policy is already good from 500k pretraining steps)

### 10.3 Entropy Regularization

Follows SAC automatic temperature tuning:
```
L(α) = E[-α · (log π_edit(â|s, a) + H_target)]
H_target = -dim(A)  # negative action dimensionality
```

---

## 11. Critic Network

### 11.1 Ensemble Implementation

```python
class QEnsemble:
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 ensemble_size=10, num_min_q=2):
        self.networks = [MLP(state_dim + action_dim, hidden_dim, 1) 
                         for _ in range(ensemble_size)]
        self.num_min_q = num_min_q
    
    def forward(self, state, action):
        """Returns Q-values from all ensemble members."""
        x = concat(state, action)
        return [net(x) for net in self.networks]
    
    def min_q(self, state, action):
        """Randomly subsample num_min_q networks, return their minimum."""
        all_q = self.forward(state, action)
        indices = random.sample(range(len(all_q)), self.num_min_q)
        selected = [all_q[i] for i in indices]
        return torch.min(torch.stack(selected), dim=0).values
```

### 11.2 Target Network Update

```python
# Polyak averaging after each critic gradient step
for param, target_param in zip(critic.parameters(), target_critic.parameters()):
    target_param.data.copy_(tau * target_param.data + (1 - tau) * param.data)
```

Where `τ = 0.005`.

---

## 12. Hyperparameters

### 12.1 Universal Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Discount factor γ | 0.99 |
| Target network τ | 0.005 |
| Q-ensemble size | 10 |
| Num Min Q | 2 |
| N action samples | 8 |
| UTD ratio G | 20 |
| Diffusion timesteps T | 10 |
| Beta schedule | Variance Preserving |
| Base policy hidden dim | 256 |
| Base policy residual blocks | 3 |
| Edit policy hidden dim | 256 |
| Edit policy hidden layers | 3 |

### 12.2 Per-Domain Hyperparameters

| Parameter | Robomimic | Adroit | Antmaze | MimicGen |
|-----------|-----------|--------|---------|----------|
| Pretraining steps | 200k | 20k | 500k | 200k |
| Edit policy dropout | None | 0.1 | None | None |
| β (online) | 0.05 | 0.7 | 0.05 | 0.05 |
| β (offline-to-online) | 0.1 | 0.7 | 0.0 | 0.05 |

### 12.3 Tuning Guidance

- **β**: Only hyperparameter that needs per-task tuning. Search over {0.05, 0.1, 0.3, 0.7}. Use smaller values when offline data is good; larger when more exploration is needed.
- **N**: Higher N may help for high-dimensional action spaces. Default 8 works well across tested domains.

---

## 13. Evaluation Protocol

### 13.1 Schedule

| Domain | Eval frequency | Eval episodes |
|--------|----------------|---------------|
| Adroit | Every 5k steps | 100 episodes |
| Antmaze | Every 5k steps | 100 episodes |
| Robomimic | Every 10k steps | 50 episodes |
| MimicGen | Every 10k steps | 50 episodes |

### 13.2 Metrics

- **Normalized Returns**: Task-specific normalization to [0, 1]
- **Adroit**: Percentage of total timesteps the task is considered solved
- **All tasks**: Sparse binary reward (1 if completed, 0 otherwise)
- **Reporting**: Mean over 3 seeds, error bars showing min and max

### 13.3 Evaluation Mode

During evaluation:
- Use the OTF policy with the **target critic**
- Disable entropy / exploration noise from the edit policy
- Alternatively, use only the base policy samples (no edits) for a pure IL evaluation

---

## 14. Full Pseudocode

### 14.1 Complete Training Loop

```python
# ============================================
# EXPO: Full Training Implementation
# ============================================

# --- Initialization ---
base_policy = DiffusionPolicy(state_dim, action_dim, T=10)
edit_policy = GaussianEditPolicy(state_dim, action_dim, beta=0.05)
critic = QEnsemble(state_dim, action_dim, ensemble_size=10)
target_critic = deepcopy(critic)
alpha = AutoTunedAlpha(target_entropy=-action_dim)

offline_buffer = load_offline_data(dataset_path)
online_buffer = ReplayBuffer(capacity=1_000_000)

optimizer_base = Adam(base_policy.parameters(), lr=3e-4)
optimizer_edit = Adam(edit_policy.parameters(), lr=3e-4)
optimizer_critic = Adam(critic.parameters(), lr=3e-4)
optimizer_alpha = Adam([alpha.log_alpha], lr=3e-4)

# --- Optional: Pretraining ---
for step in range(pretraining_steps):
    batch = offline_buffer.sample(256)
    loss_il = base_policy.diffusion_loss(batch['state'], batch['action'])
    optimizer_base.zero_grad()
    loss_il.backward()
    optimizer_base.step()

# --- Online Training ---
state = env.reset()
for step in range(total_env_steps):
    
    # === Action Selection ===
    with torch.no_grad():
        action = otf_action_selection(
            state, base_policy, edit_policy, target_critic,
            N=8, beta=edit_policy.beta
        )
    
    next_state, reward, done, info = env.step(action)
    online_buffer.add(state, action, reward, next_state, done)
    state = next_state if not done else env.reset()
    
    # === Gradient Updates (UTD = 20) ===
    for g in range(20):
        # Sample 50/50 from offline and online
        batch_offline = offline_buffer.sample(128)
        batch_online = online_buffer.sample(128)
        batch = merge_batches(batch_offline, batch_online)
        
        s, a, r, s_next, done_mask = unpack(batch)
        
        # --- Critic Update ---
        with torch.no_grad():
            next_action = otf_action_selection(
                s_next, base_policy, edit_policy, target_critic,
                N=8, beta=edit_policy.beta
            )
            target_q = target_critic.min_q(s_next, next_action)
            td_target = r + (1 - done_mask) * gamma * target_q
        
        current_q_list = critic.forward(s, a)
        critic_loss = sum(F.mse_loss(q, td_target) for q in current_q_list)
        
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()
        
        # --- Target Update ---
        soft_update(target_critic, critic, tau=0.005)
    
    # --- Base Policy Update (on last batch) ---
    il_loss = base_policy.diffusion_loss(s, a)  # using replay data
    optimizer_base.zero_grad()
    il_loss.backward()
    optimizer_base.step()
    
    # --- Edit Policy Update (on last batch) ---
    with torch.no_grad():
        base_actions = base_policy.sample(s)
    edits, log_probs = edit_policy(s, base_actions)
    edited_actions = (base_actions + edits).clamp(-1, 1)
    
    q_edited = critic.min_q(s, edited_actions)
    edit_loss = -(q_edited - alpha.value * log_probs).mean()
    
    optimizer_edit.zero_grad()
    edit_loss.backward()
    optimizer_edit.step()
    
    # --- Alpha Update ---
    alpha_loss = -(alpha.log_alpha * (log_probs.detach() + alpha.target_entropy)).mean()
    optimizer_alpha.zero_grad()
    alpha_loss.backward()
    optimizer_alpha.step()
```

---

## 15. Dependency Map

### 15.1 Required Libraries

```
torch >= 2.0
numpy
gym / gymnasium
d4rl (for benchmark environments)
robosuite (for Robomimic/MimicGen)
mimicgen
mujoco
wandb or tensorboard (logging)
```

### 15.2 Module Structure

```
expo/
├── models/
│   ├── diffusion_policy.py      # DDPM base policy
│   ├── edit_policy.py           # Gaussian edit policy
│   ├── critic.py                # Q-ensemble
│   └── networks.py              # Shared MLP, ResBlock definitions
├── agents/
│   ├── expo_agent.py            # Main EXPO agent with OTF extraction
│   └── sac_utils.py             # Auto-tuned alpha, soft update
├── buffers/
│   ├── replay_buffer.py         # Standard replay buffer
│   └── mixed_buffer.py          # 50/50 offline-online sampling
├── envs/
│   ├── antmaze.py               # D4RL Antmaze wrapper
│   ├── adroit.py                # D4RL Adroit wrapper
│   ├── robomimic.py             # Robomimic wrapper
│   └── mimicgen.py              # MimicGen wrapper
├── training/
│   ├── pretrain.py              # IL pretraining script
│   ├── online.py                # Online RL loop
│   └── eval.py                  # Evaluation utilities
├── configs/
│   ├── robomimic.yaml
│   ├── adroit.yaml
│   ├── antmaze.yaml
│   └── mimicgen.yaml
└── main.py                      # Entry point
```

### 15.3 Key Implementation Checklist

- [ ] DDPM forward/reverse process with VP schedule (T=10)
- [ ] Residual block MLP for denoising network
- [ ] Gaussian edit policy with tanh squashing to [-β, β]
- [ ] Q-ensemble with random subsampling (10 networks, min over 2)
- [ ] OTF action selection (N=8 samples, 2N candidates, argmax Q)
- [ ] TD learning with OTF target actions
- [ ] SAC-style entropy tuning for edit policy
- [ ] Mixed replay buffer (offline + online)
- [ ] Soft target network updates (τ=0.005)
- [ ] High UTD ratio training (G=20)
- [ ] Base policy IL loss on full replay buffer
- [ ] Per-domain β tuning
- [ ] Evaluation protocol with proper frequency and episodes
