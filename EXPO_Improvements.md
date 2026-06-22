# EXPO: Suggested Improvements & Extensions

## Table of Contents

1. [Computational Efficiency](#1-computational-efficiency)
2. [Edit Policy Enhancements](#2-edit-policy-enhancements)
3. [Base Policy Improvements](#3-base-policy-improvements)
4. [Critic & Value Learning](#4-critic--value-learning)
5. [Exploration Strategy](#5-exploration-strategy)
6. [Data & Buffer Management](#6-data--buffer-management)
7. [Architecture Modernization](#7-architecture-modernization)
8. [Training Stability](#8-training-stability)
9. [Scalability & Multi-Task](#9-scalability--multi-task)
10. [Practical Deployment](#10-practical-deployment)

Each improvement includes: motivation, design, and implementation pseudocode.

---

## 1. Computational Efficiency

### 1.1 Cached Base Policy Sampling

**Problem**: The OTF extraction requires sampling N actions from the diffusion base policy for every training batch element during TD backup. With batch size 256, UTD 20, and N=8, that is 256 × 8 × 10 (diffusion steps) = 20,480 denoising forward passes per environment step just for the TD target — and this happens 20 times per step.

**Solution**: Cache base policy samples. Since the base policy updates only once per env step (not per UTD step), its samples remain valid across all G critic updates within the same step.

```python
# Improvement: Cache OTF actions across UTD steps

class CachedOTFExtractor:
    def __init__(self):
        self.cache = {}
        self.cache_step = -1
    
    def get_otf_actions(self, states, base_policy, edit_policy, critic, 
                        N, beta, current_step):
        state_key = hash(states.data_ptr())
        
        if self.cache_step != current_step or state_key not in self.cache:
            # Only recompute base actions when policy has updated
            with torch.no_grad():
                base_actions = self._sample_base_batch(states, base_policy, N)
            self.cache[state_key] = base_actions
            self.cache_step = current_step
        
        base_actions = self.cache[state_key]
        
        # Edit policy is lightweight — always recompute edits
        # (edit policy also updates once per step, but edits are cheap)
        edits, _ = edit_policy(states_repeated, base_actions)
        edited_actions = (base_actions + edits.clamp(-beta, beta)).clamp(-1, 1)
        
        all_actions = torch.cat([base_actions, edited_actions], dim=1)
        q_vals = critic.min_q(states_expanded, all_actions.reshape(-1, action_dim))
        q_vals = q_vals.reshape(batch_size, 2 * N)
        
        best_idx = q_vals.argmax(dim=1)
        return all_actions[torch.arange(batch_size), best_idx]
```

**Expected Speedup**: ~10-15x reduction in diffusion sampling cost during training, since base actions are sampled once instead of G=20 times.

---

### 1.2 Distilled One-Step Base Policy for TD Backup

**Problem**: Even with caching, diffusion sampling for rollouts is slow (T=10 denoising steps per action).

**Solution**: Maintain a one-step distilled version of the base policy for fast inference during TD backup, while keeping the full diffusion policy for rollout diversity.

```python
class DistilledBasePolicy(nn.Module):
    """One-step policy distilled from the diffusion base policy."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )
    
    def forward(self, state, noise=None):
        mean = self.net(state)
        if noise is None:
            noise = torch.randn_like(mean) * 0.1
        return (mean + noise).clamp(-1, 1)

# Distillation loss — run periodically (every K env steps)
def distill_base_policy(diffusion_policy, distilled_policy, states, 
                        num_samples=64):
    with torch.no_grad():
        # Generate diverse targets from diffusion
        targets = []
        for _ in range(num_samples):
            targets.append(diffusion_policy.sample(states))
        targets = torch.stack(targets, dim=1)  # [B, num_samples, A]
    
    # Train distilled policy to match the mean
    predicted = distilled_policy(states).unsqueeze(1)  # [B, 1, A]
    loss = ((predicted - targets) ** 2).mean()
    return loss
```

---

### 1.3 Parallel Batch OTF Extraction

**Problem**: Sequential sampling and evaluation of 2N candidates per state.

**Solution**: Fully vectorized implementation that processes all candidates in a single batched forward pass.

```python
def vectorized_otf(states, base_policy, edit_policy, critic, N, beta):
    """Fully vectorized OTF — single forward pass per network."""
    B, S = states.shape
    A = edit_policy.action_dim
    
    # [B, N, S] → [B*N, S]
    s_exp = states.unsqueeze(1).expand(B, N, S).reshape(B * N, S)
    
    # Single batched diffusion call
    base_acts = base_policy.sample(s_exp)           # [B*N, A]
    edits, _ = edit_policy(s_exp, base_acts)         # [B*N, A]
    edited_acts = (base_acts + edits.clamp(-beta, beta)).clamp(-1, 1)
    
    # Stack all 2N candidates: [B*2N, A]
    all_acts = torch.cat([base_acts, edited_acts], dim=0)
    all_states = torch.cat([s_exp, s_exp], dim=0)    # [B*2N, S]
    
    # Single batched critic call
    q = critic.min_q(all_states, all_acts)           # [B*2N, 1]
    q = q.reshape(2, B, N).permute(1, 2, 0).reshape(B, 2 * N)  # [B, 2N]
    
    all_acts = all_acts.reshape(2, B, N, A).permute(1, 2, 0, 3).reshape(B, 2*N, A)
    best_idx = q.argmax(dim=1)
    return all_acts[torch.arange(B), best_idx]
```

---

## 2. Edit Policy Enhancements

### 2.1 State-Dependent β (Adaptive Edit Distance)

**Problem**: Fixed β across all states is suboptimal. Some states need large edits (unexplored regions), others need fine-grained refinement (near-optimal states).

**Solution**: Learn a state-dependent β that adapts the edit magnitude based on the Q-landscape curvature.

```python
class AdaptiveEditPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 beta_min=0.01, beta_max=0.5):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # β prediction head
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()  # outputs in [0, 1]
        )
    
    def forward(self, state, base_action):
        x = torch.cat([state, base_action], dim=-1)
        features = self.backbone(x)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(-20, 2)
        
        # Adaptive β per state
        beta_ratio = self.beta_head(features.detach())  # detach to avoid Q gradient
        beta = self.beta_min + (self.beta_max - self.beta_min) * beta_ratio
        
        # Sample and scale
        std = log_std.exp()
        raw = mean + std * torch.randn_like(std)
        edit = beta * torch.tanh(raw)
        
        return edit, self._log_prob(raw, mean, std, edit, beta)
    
    def _log_prob(self, raw, mean, std, edit, beta):
        normal_lp = -0.5 * ((raw - mean) / std).pow(2) - log_std - 0.5 * np.log(2 * np.pi)
        tanh_correction = torch.log(1 - (edit / beta).pow(2) + 1e-6)
        return (normal_lp - tanh_correction).sum(dim=-1, keepdim=True)
```

---

### 2.2 Multi-Step Edit Refinement

**Problem**: A single edit step may not be sufficient to find the optimal action perturbation, especially in high-dimensional action spaces.

**Solution**: Apply K iterative refinement steps, where each step conditions on the previously edited action.

```python
def iterative_edit(state, base_action, edit_policy, critic, K=3, beta=0.05):
    """Apply K sequential edits, each refining the previous result."""
    current_action = base_action.clone()
    
    for k in range(K):
        edit, _ = edit_policy(state, current_action)
        # Decay edit magnitude over iterations for convergence
        decay = beta * (0.5 ** k)
        edit = edit.clamp(-decay, decay)
        current_action = (current_action + edit).clamp(-1, 1)
    
    return current_action
```

This is complementary to the OTF extraction — you can apply iterative edits to each of the N base samples before the argmax selection.

---

### 2.3 Mixture of Edits

**Problem**: A single Gaussian edit policy can only capture unimodal perturbations per base action.

**Solution**: Use a Gaussian Mixture edit policy to capture multiple promising edit directions.

```python
class MixtureEditPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 n_components=3, beta=0.05):
        super().__init__()
        self.n_components = n_components
        self.beta = beta
        
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Per-component outputs
        self.mean_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(n_components)
        ])
        self.log_std_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(n_components)
        ])
        # Mixture weights
        self.logits_head = nn.Linear(hidden_dim, n_components)
    
    def forward(self, state, base_action):
        features = self.backbone(torch.cat([state, base_action], dim=-1))
        logits = self.logits_head(features)
        
        # Sample component
        comp_idx = Categorical(logits=logits).sample()
        
        mean = self.mean_heads[comp_idx](features)
        log_std = self.log_std_heads[comp_idx](features).clamp(-20, 2)
        
        raw = mean + log_std.exp() * torch.randn_like(mean)
        edit = self.beta * torch.tanh(raw)
        
        return edit, self._mixture_log_prob(features, edit)
```

---

## 3. Base Policy Improvements

### 3.1 Flow Matching Instead of DDPM

**Problem**: DDPM with T=10 steps is slower than necessary and the VP schedule may not be optimal for all domains.

**Solution**: Replace the base policy with a Conditional Flow Matching (CFM) model that can sample in fewer steps with better quality.

```python
class FlowMatchingPolicy(nn.Module):
    """Conditional Flow Matching base policy — typically 5-8 steps."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_blocks=3):
        super().__init__()
        self.velocity_net = ResidualMLP(
            input_dim=action_dim + state_dim + 1,  # +1 for time
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_blocks=num_blocks
        )
    
    def velocity(self, x_t, state, t):
        """Predict velocity field v(x_t, s, t)."""
        t_embed = t.unsqueeze(-1) if t.dim() == 1 else t
        inp = torch.cat([x_t, state, t_embed], dim=-1)
        return self.velocity_net(inp)
    
    def training_loss(self, state, action):
        """Conditional flow matching loss."""
        t = torch.rand(state.shape[0], device=state.device)
        noise = torch.randn_like(action)
        
        # Optimal transport interpolant
        x_t = (1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * action
        target_velocity = action - noise  # OT velocity
        
        predicted = self.velocity(x_t, state, t)
        return F.mse_loss(predicted, target_velocity)
    
    def sample(self, state, num_steps=5):
        """Euler integration for sampling."""
        x = torch.randn(state.shape[0], self.action_dim, device=state.device)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((state.shape[0],), i * dt, device=state.device)
            v = self.velocity(x, state, t)
            x = x + v * dt
        
        return x.clamp(-1, 1)
```

**Benefit**: 2x fewer denoising steps (5 vs 10), better interpolation quality via optimal transport.

---

### 3.2 Weighted Imitation Learning for Base Policy

**Problem**: The base policy treats all data equally in the IL loss, but online-collected high-reward data should be emphasized.

**Solution**: Weight the IL loss by advantage or Q-value, without backpropagating Q gradients through the diffusion chain.

```python
def weighted_diffusion_loss(base_policy, critic, states, actions, 
                            temperature=1.0):
    """Advantage-weighted diffusion loss — still pure supervision."""
    with torch.no_grad():
        q_values = critic.min_q(states, actions)
        # Normalize advantages within batch for stability
        advantages = q_values - q_values.mean()
        weights = torch.softmax(advantages / temperature, dim=0)
        weights = weights * len(weights)  # rescale so mean weight ≈ 1
    
    # Standard diffusion loss, but weighted
    t = torch.randint(1, T + 1, (states.shape[0],))
    noise = torch.randn_like(actions)
    alpha_bar = get_alpha_bar(t)
    
    noised = alpha_bar.sqrt() * actions + (1 - alpha_bar).sqrt() * noise
    predicted_noise = base_policy.denoise(noised, states, t)
    
    per_sample_loss = ((noise - predicted_noise) ** 2).mean(dim=-1)
    weighted_loss = (weights.squeeze() * per_sample_loss).mean()
    
    return weighted_loss
```

**Key**: Weights are detached — no Q gradients flow through diffusion. This biases the base policy toward high-value actions while maintaining training stability.

---

## 4. Critic & Value Learning

### 4.1 Layer Normalization in Critic

**Problem**: High UTD ratios (G=20) can cause critic feature rank collapse and loss of plasticity.

**Solution**: Add LayerNorm after the first linear layer of each Q-network, following recent findings on maintaining network plasticity.

```python
class PlasticQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Critical for high UTD
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))
```

---

### 4.2 Softmax OTF Instead of Hard Argmax

**Problem**: Hard argmax selection is non-smooth and can be noisy when Q-values are close.

**Solution**: Use a softmax-weighted combination (or Boltzmann sampling) over candidates for smoother action selection.

```python
def softmax_otf(states, all_actions, q_values, temperature=0.1, mode='sample'):
    """
    Soft OTF extraction.
    
    Args:
        states: [B, S]
        all_actions: [B, 2N, A]
        q_values: [B, 2N]
        temperature: softmax temperature (lower = closer to argmax)
        mode: 'sample' for stochastic, 'weighted' for deterministic weighted mean
    """
    weights = F.softmax(q_values / temperature, dim=1)  # [B, 2N]
    
    if mode == 'sample':
        # Boltzmann sampling
        idx = Categorical(probs=weights).sample()  # [B]
        return all_actions[torch.arange(len(idx)), idx]
    elif mode == 'weighted':
        # Weighted average (smooth, differentiable)
        return (weights.unsqueeze(-1) * all_actions).sum(dim=1)
```

**Use cases**:
- Rollout: Use `mode='sample'` with moderate temperature for exploration
- TD backup: Use `mode='weighted'` or hard argmax for stability
- Evaluation: Use hard argmax

---

### 4.3 N-Step Returns

**Problem**: Single-step TD can be slow to propagate reward signal, especially in sparse-reward tasks.

**Solution**: Combine 1-step and n-step returns for faster credit assignment.

```python
class NStepBuffer:
    """Wrapper that computes n-step returns on the fly."""
    
    def __init__(self, base_buffer, n=3, gamma=0.99):
        self.buffer = base_buffer
        self.n = n
        self.gamma = gamma
        self.pending = deque(maxlen=n)
    
    def add(self, state, action, reward, next_state, done):
        self.pending.append((state, action, reward, next_state, done))
        
        if len(self.pending) == self.n or done:
            # Compute n-step return
            n_step_return = 0
            for i, (s, a, r, ns, d) in enumerate(reversed(list(self.pending))):
                n_step_return = r + self.gamma * n_step_return * (1 - d)
            
            first = self.pending[0]
            last = self.pending[-1]
            self.buffer.add(
                first[0], first[1],  # first state, action
                n_step_return,        # accumulated reward
                last[3],              # last next_state
                last[4],              # last done
                self.n                # store actual n for variable-length
            )
            
            if not done:
                self.pending.popleft()
            else:
                self.pending.clear()
    
    def sample(self, batch_size):
        batch = self.buffer.sample(batch_size)
        # Adjust discount: γ^n instead of γ
        batch['discount'] = self.gamma ** batch['n_steps']
        return batch
```

---

## 5. Exploration Strategy

### 5.1 Curiosity-Driven Edit Bonus

**Problem**: Entropy regularization alone may not provide sufficient directed exploration in very sparse reward environments.

**Solution**: Add a curiosity bonus (based on prediction error) to the edit policy's reward signal.

```python
class CuriosityModule(nn.Module):
    """Random Network Distillation (RND) for exploration bonus."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + action_dim
        
        # Fixed random target network
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        for p in self.target.parameters():
            p.requires_grad = False
        
        # Trainable predictor
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
    
    def bonus(self, state, action):
        x = torch.cat([state, action], dim=-1)
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        return ((target_feat - pred_feat) ** 2).mean(dim=-1, keepdim=True)
    
    def update(self, state, action):
        loss = self.bonus(state, action).mean()
        return loss

# Modified edit policy loss with curiosity
def edit_loss_with_curiosity(edit_policy, critic, curiosity, states, 
                              base_actions, alpha, curiosity_coeff=0.1):
    edits, log_probs = edit_policy(states, base_actions)
    edited = (base_actions + edits).clamp(-1, 1)
    
    q_value = critic.min_q(states, edited)
    exploration_bonus = curiosity.bonus(states, edited)
    
    loss = -(q_value + curiosity_coeff * exploration_bonus 
             - alpha * log_probs).mean()
    return loss
```

---

### 5.2 Progressive β Schedule

**Problem**: Fixed β means the edit magnitude is static throughout training. Early training needs more exploration; later training needs precision.

**Solution**: Anneal β from a large initial value to a small final value.

```python
class ProgressiveBeta:
    def __init__(self, beta_start=0.5, beta_end=0.05, 
                 warmup_steps=50_000, decay_steps=200_000):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
    
    def __call__(self, step):
        if step < self.warmup_steps:
            return self.beta_start
        
        progress = min(1.0, (step - self.warmup_steps) / self.decay_steps)
        # Cosine decay
        beta = self.beta_end + 0.5 * (self.beta_start - self.beta_end) * \
               (1 + np.cos(np.pi * progress))
        return beta
```

---

## 6. Data & Buffer Management

### 6.1 Prioritized Experience Replay

**Problem**: Uniform sampling from the replay buffer doesn't prioritize informative transitions.

**Solution**: Use TD-error-based priorities for the online buffer while keeping uniform sampling for the offline buffer.

```python
class PrioritizedMixedBuffer:
    def __init__(self, offline_data, online_capacity, alpha=0.6, 
                 online_ratio=0.5):
        self.offline = UniformBuffer(offline_data)
        self.online = PrioritizedBuffer(online_capacity, alpha=alpha)
        self.online_ratio = online_ratio
    
    def sample(self, batch_size):
        n_online = int(batch_size * self.online_ratio)
        n_offline = batch_size - n_online
        
        offline_batch = self.offline.sample(n_offline)
        online_batch, online_indices, online_weights = \
            self.online.sample(n_online)
        
        # Merge batches
        batch = merge(offline_batch, online_batch)
        batch['weights'] = torch.cat([
            torch.ones(n_offline),
            online_weights
        ])
        batch['priority_indices'] = online_indices
        return batch
    
    def update_priorities(self, indices, td_errors):
        self.online.update_priorities(indices, td_errors.abs() + 1e-6)
```

---

### 6.2 Adaptive Online-Offline Sampling Ratio

**Problem**: Fixed 50/50 ratio may be suboptimal. Early in training, offline data is more valuable; later, online data becomes more relevant.

**Solution**: Dynamically adjust the ratio based on buffer sizes and training progress.

```python
class AdaptiveSamplingRatio:
    def __init__(self, initial_offline_ratio=0.8, min_offline_ratio=0.2,
                 transition_steps=100_000):
        self.initial = initial_offline_ratio
        self.minimum = min_offline_ratio
        self.transition_steps = transition_steps
    
    def get_ratio(self, step, online_buffer_size, offline_buffer_size):
        # Linear decay of offline ratio
        progress = min(1.0, step / self.transition_steps)
        offline_ratio = self.initial - (self.initial - self.minimum) * progress
        
        # Also consider relative buffer sizes
        if online_buffer_size < 1000:
            offline_ratio = max(offline_ratio, 0.9)  # lean on offline early
        
        return offline_ratio
```

---

## 7. Architecture Modernization

### 7.1 Transformer-Based Denoising Network

**Problem**: Residual MLPs may not capture complex temporal/spatial dependencies in high-dimensional observations.

**Solution**: Use a small transformer as the denoising backbone for the base policy.

```python
class TransformerDenoiser(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 n_heads=4, n_layers=3):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_embed = SinusoidalEmbedding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.0,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.output_proj = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, noised_action, state, timestep):
        s_tok = self.state_proj(state).unsqueeze(1)        # [B, 1, H]
        a_tok = self.action_proj(noised_action).unsqueeze(1)  # [B, 1, H]
        t_tok = self.time_embed(timestep).unsqueeze(1)      # [B, 1, H]
        
        tokens = torch.cat([t_tok, s_tok, a_tok], dim=1)   # [B, 3, H]
        out = self.transformer(tokens)
        
        # Use action token output
        return self.output_proj(out[:, 2, :])  # [B, action_dim]
```

---

### 7.2 Spectral Normalization for Critic Stability

**Problem**: Q-ensemble with high UTD can produce overestimated or unstable Q-values.

**Solution**: Apply spectral normalization to critic weights for Lipschitz-constrained value functions.

```python
from torch.nn.utils.parametrizations import spectral_norm

class SpectralNormQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(state_dim + action_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # No SN on last layer
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))
```

---

## 8. Training Stability

### 8.1 Gradient Clipping for Edit Policy

**Problem**: Large Q-value gradients can cause unstable edit policy updates, especially early in training when the Q-function is not well calibrated.

**Solution**: Clip gradients by norm for the edit policy.

```python
# In training loop, after edit_loss.backward():
torch.nn.utils.clip_grad_norm_(edit_policy.parameters(), max_norm=1.0)
optimizer_edit.step()
```

---

### 8.2 Delayed Edit Policy Start

**Problem**: The edit policy starts training immediately, but the Q-function is randomly initialized and provides noisy gradients.

**Solution**: Delay edit policy training until the critic has stabilized.

```python
# In training loop:
if step > critic_warmup_steps:  # e.g., 5000 steps
    # Train edit policy
    edit_loss = compute_edit_loss(...)
    optimizer_edit.zero_grad()
    edit_loss.backward()
    optimizer_edit.step()
else:
    # During warmup, OTF just selects best base action (no edits)
    pass
```

---

### 8.3 EMA Base Policy for Stable OTF

**Problem**: The base policy is updated every step, which can cause the OTF policy distribution to shift rapidly.

**Solution**: Maintain an EMA copy of the base policy for use in OTF extraction, similar to target critic.

```python
class EMABasePolicy:
    def __init__(self, base_policy, tau=0.005):
        self.ema_policy = deepcopy(base_policy)
        self.tau = tau
    
    def update(self, base_policy):
        for p_ema, p in zip(self.ema_policy.parameters(), 
                            base_policy.parameters()):
            p_ema.data.copy_(self.tau * p_ema.data + (1 - self.tau) * p.data)
    
    def sample(self, state):
        with torch.no_grad():
            return self.ema_policy.sample(state)
```

---

## 9. Scalability & Multi-Task

### 9.1 Multi-Task EXPO with Shared Base Policy

**Problem**: EXPO is evaluated per-task. Scaling to multi-task settings with a shared base policy would enable broader pre-training.

**Solution**: Shared base policy with task-conditioned edit policies.

```python
class TaskConditionedEditPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, task_dim, hidden_dim=256):
        super().__init__()
        self.task_embed = nn.Embedding(num_tasks, task_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state, base_action, task_id):
        task_feat = self.task_embed(task_id)
        x = torch.cat([state, base_action, task_feat], dim=-1)
        features = self.net(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(-20, 2)
        return self._sample(mean, log_std)
```

---

### 9.2 Hierarchical Action Edits

**Problem**: For high-dimensional action spaces (e.g., 28-DoF Adroit hand), a single flat edit is challenging.

**Solution**: Decompose the action space into groups and apply group-specific edit heads.

```python
class HierarchicalEditPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, groups, hidden_dim=256, beta=0.1):
        """
        groups: list of (start_idx, end_idx) tuples defining action groups
        e.g., for a robot arm: [(0,7), (7,14), (14,21), (21,28)]
        """
        super().__init__()
        self.groups = groups
        self.beta = beta
        
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        
        self.group_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128), nn.ReLU(),
                nn.Linear(128, 2 * (end - start))  # mean + log_std
            )
            for start, end in groups
        ])
    
    def forward(self, state, base_action):
        features = self.shared_backbone(torch.cat([state, base_action], -1))
        
        edits = []
        log_probs = []
        for i, (start, end) in enumerate(self.groups):
            out = self.group_heads[i](features)
            dim = end - start
            mean, log_std = out[:, :dim], out[:, dim:].clamp(-20, 2)
            
            std = log_std.exp()
            raw = mean + std * torch.randn_like(std)
            edit = self.beta * torch.tanh(raw)
            
            edits.append(edit)
            lp = Normal(mean, std).log_prob(raw) - torch.log(1 - (edit/self.beta)**2 + 1e-6)
            log_probs.append(lp.sum(dim=-1))
        
        full_edit = torch.cat(edits, dim=-1)
        total_log_prob = sum(log_probs).unsqueeze(-1)
        return full_edit, total_log_prob
```

---

## 10. Practical Deployment

### 10.1 Online Evaluation with No-Edit Fallback

**Problem**: During deployment, the edit policy's quality depends on the Q-function, which may be miscalibrated.

**Solution**: At deployment time, compare the edited action's Q-value against the base action's Q-value and only apply the edit if it provides a meaningful improvement.

```python
def safe_otf_action(state, base_policy, edit_policy, critic, 
                     N=8, beta=0.05, min_improvement=0.01):
    """Conservative OTF that only edits when confident."""
    base_actions = base_policy.sample(state.expand(N, -1))
    edits, _ = edit_policy(state.expand(N, -1), base_actions)
    edited = (base_actions + edits.clamp(-beta, beta)).clamp(-1, 1)
    
    q_base = critic.min_q(state.expand(N, -1), base_actions)
    q_edited = critic.min_q(state.expand(N, -1), edited)
    
    # Only use edited actions that are meaningfully better
    use_edit = (q_edited - q_base) > min_improvement
    final_actions = torch.where(use_edit, edited, base_actions)
    
    q_final = torch.where(use_edit, q_edited, q_base)
    best_idx = q_final.argmax(dim=0)
    
    return final_actions[best_idx]
```

---

### 10.2 Logging and Diagnostics

**Problem**: EXPO has many moving parts — diagnosing failures requires tracking the right metrics.

**Solution**: Comprehensive logging suite.

```python
class EXPOLogger:
    """Track key diagnostic metrics for EXPO training."""
    
    def log_step(self, step, metrics):
        log = {
            # === Performance ===
            'eval/success_rate': metrics['success_rate'],
            'eval/return': metrics['return'],
            
            # === Base Policy ===
            'base/il_loss': metrics['il_loss'],
            'base/action_std': metrics['base_action_std'],
            
            # === Edit Policy ===
            'edit/loss': metrics['edit_loss'],
            'edit/mean_edit_magnitude': metrics['edit_magnitude'],
            'edit/entropy': metrics['edit_entropy'],
            'edit/alpha': metrics['alpha'],
            
            # === Critic ===
            'critic/loss': metrics['critic_loss'],
            'critic/mean_q': metrics['mean_q'],
            'critic/max_q': metrics['max_q'],
            'critic/td_error': metrics['td_error'],
            'critic/q_std_across_ensemble': metrics['q_ensemble_std'],
            
            # === OTF Diagnostics ===
            'otf/fraction_edited_selected': metrics['frac_edited'],
            'otf/q_improvement_from_edit': metrics['q_delta_edit'],
            'otf/best_q_vs_mean_q': metrics['q_gap'],
            
            # === Buffer ===
            'buffer/online_size': metrics['online_buffer_size'],
            'buffer/offline_ratio': metrics['offline_sample_ratio'],
        }
        wandb.log(log, step=step)
```

Key diagnostic signals:
- `otf/fraction_edited_selected`: Should be >0 (edits are useful) but <1 (base still contributes). If always 1, β may be too large. If always 0, edit policy isn't learning.
- `edit/mean_edit_magnitude`: Should be meaningfully non-zero but bounded by β.
- `critic/q_std_across_ensemble`: If too high, Q-estimates are unreliable.
- `otf/q_improvement_from_edit`: Should be positive on average — edits should improve Q-values.

---

### 10.3 Checkpoint and Resume Strategy

```python
def save_checkpoint(path, step, base_policy, edit_policy, critic, 
                    target_critic, alpha, optimizers, buffer_stats):
    torch.save({
        'step': step,
        'base_policy': base_policy.state_dict(),
        'edit_policy': edit_policy.state_dict(),
        'critic': critic.state_dict(),
        'target_critic': target_critic.state_dict(),
        'log_alpha': alpha.log_alpha,
        'optimizer_base': optimizers['base'].state_dict(),
        'optimizer_edit': optimizers['edit'].state_dict(),
        'optimizer_critic': optimizers['critic'].state_dict(),
        'optimizer_alpha': optimizers['alpha'].state_dict(),
        'buffer_stats': buffer_stats,
    }, path)

def load_checkpoint(path, base_policy, edit_policy, critic, 
                    target_critic, alpha, optimizers):
    ckpt = torch.load(path)
    base_policy.load_state_dict(ckpt['base_policy'])
    edit_policy.load_state_dict(ckpt['edit_policy'])
    critic.load_state_dict(ckpt['critic'])
    target_critic.load_state_dict(ckpt['target_critic'])
    alpha.log_alpha.data = ckpt['log_alpha']
    for key in optimizers:
        optimizers[key].load_state_dict(ckpt[f'optimizer_{key}'])
    return ckpt['step']
```

---

## Summary: Priority-Ranked Improvements

| Priority | Improvement | Expected Impact | Effort |
|----------|-------------|-----------------|--------|
| 🔴 High | Cached base policy sampling (1.1) | 10-15x training speedup | Low |
| 🔴 High | LayerNorm in critic (4.1) | Better stability at UTD=20 | Low |
| 🔴 High | Gradient clipping for edit policy (8.1) | Prevents early training collapse | Low |
| 🟡 Medium | Progressive β schedule (5.2) | Better explore→exploit transition | Low |
| 🟡 Medium | Weighted IL for base policy (3.2) | Faster base policy adaptation | Medium |
| 🟡 Medium | N-step returns (4.3) | Faster credit assignment in sparse reward | Medium |
| 🟡 Medium | Softmax OTF (4.2) | Smoother action selection | Low |
| 🟡 Medium | Comprehensive logging (10.2) | Much easier debugging | Medium |
| 🟢 Low | State-dependent β (2.1) | Per-state exploration tuning | Medium |
| 🟢 Low | Flow matching base policy (3.1) | Faster sampling, better quality | High |
| 🟢 Low | Multi-step edits (2.2) | Better refinement in high-dim | Low |
| 🟢 Low | Curiosity bonus (5.1) | Directed exploration | Medium |
| 🟢 Low | Transformer denoiser (7.1) | Better representational capacity | High |
| 🟢 Low | Hierarchical edits (9.2) | High-dim action space handling | Medium |
