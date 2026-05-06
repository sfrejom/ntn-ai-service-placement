"""DRL-based placement agent and PPO trainer.

The agent autoregressively places one microservice at a time. For each round
(one environment step), the trajectory is:

    1. Encode (node_features, svc_features) once.
    2. Loop over services in a fixed (priority-sorted) order:
        a. Build the per-service feasibility mask (latency + capacity + load
           summary built from earlier placements in this round).
        b. Compute placement logits via the cross-attention scorer.
        c. Sample a node action; remember log-prob and entropy.
        d. Update the running load summary.
    3. Apply the placement, observe reward and StepInfo, push transition
       into a rollout buffer.

PPO is implemented from scratch (clipped surrogate, GAE, value-loss clipping).
The trainer is small enough to run on CPU within minutes for the experiment
sizes we care about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from simulator.environment import NTNEnvironment, Observation

from .baselines import PlacementAgent
from .policy import PlacementPolicy


@dataclass
class Transition:
    obs_node: torch.Tensor       # (N, F_n)
    obs_svc: torch.Tensor        # (M, F_s)
    feas_mask_per_step: torch.Tensor  # (M, N) feasibility per service
    load_summaries: torch.Tensor  # (M, 3) per-service running load summary
    actions: torch.Tensor        # (M,) selected node index for each service
    log_probs: torch.Tensor      # (M,)
    value: torch.Tensor          # scalar
    reward: float
    done: bool


@dataclass
class TrainConfig:
    total_episodes: int = 400
    rollout_episodes: int = 4
    epochs_per_update: int = 4
    minibatch_size: int = 32
    learning_rate: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.95
    gae_lambda: float = 0.9
    embed_dim: int = 64
    n_heads: int = 4
    grad_clip: float = 0.5
    seed: int = 0


class DRLPlacementAgent(PlacementAgent):
    """DRL-based placement policy. Trained with `train(...)`. After training,
    `place(...)` performs greedy (argmax) placement; pass `stochastic=True`
    in the constructor to sample instead."""

    name = "drl"

    def __init__(
        self,
        node_feature_dim: int,
        svc_feature_dim: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        seed: int = 0,
        stochastic_eval: bool = False,
        device: Optional[str] = None,
    ) -> None:
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = PlacementPolicy(
            node_in_dim=node_feature_dim,
            svc_in_dim=svc_feature_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
        ).to(self.device)
        self.stochastic_eval = stochastic_eval
        self.feature_dim_node = node_feature_dim
        self.feature_dim_svc = svc_feature_dim

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _service_priority_order(self, env: NTNEnvironment) -> List[int]:
        # Tightest-latency first. Ties broken by stateful, then CPU demand.
        order = list(range(len(env.services)))
        order.sort(
            key=lambda j: (
                env.services[j].max_latency_ms,
                -int(env.services[j].is_stateful),
                -env.services[j].cpu_demand,
            )
        )
        return order

    def _capacity_feasible(
        self,
        env: NTNEnvironment,
        cpu_used: np.ndarray,
        mem_used: np.ndarray,
        svc_idx: int,
    ) -> np.ndarray:
        svc = env.services[svc_idx]
        N = len(env.nodes)
        feas = np.zeros(N, dtype=bool)
        latency_ok = env.feasibility_mask()[:, svc_idx]
        for i, node in enumerate(env.nodes):
            if not node.active or not latency_ok[i]:
                continue
            if cpu_used[i] + svc.cpu_demand > node.cpu_capacity:
                continue
            if mem_used[i] + svc.mem_demand > node.mem_capacity:
                continue
            # Coverage
            ground = np.linalg.norm(env.users[:, :2] - node.position[None, :2], axis=1)
            if not (ground <= node.coverage_radius_km).any():
                continue
            feas[i] = True
        return feas

    @torch.no_grad()
    def place(self, env: NTNEnvironment, obs: Observation) -> np.ndarray:
        N = len(env.nodes)
        M = len(env.services)
        node_t = torch.from_numpy(obs.node_features).float().unsqueeze(0).to(self.device)
        svc_t = torch.from_numpy(obs.service_features).float().unsqueeze(0).to(self.device)
        ctx, svc_embed = self.policy.encode(node_t, svc_t)
        X = np.zeros((N, M), dtype=np.int32)
        cpu_used = np.zeros(N)
        mem_used = np.zeros(N)
        order = self._service_priority_order(env)
        for j in order:
            feas = self._capacity_feasible(env, cpu_used, mem_used, j)
            if not feas.any():
                continue
            feas_t = torch.from_numpy(feas).bool().unsqueeze(0).to(self.device)
            load_summary = torch.tensor(
                [
                    [
                        cpu_used.sum() / max(1.0, sum(n.cpu_capacity for n in env.nodes)),
                        mem_used.sum() / max(1.0, sum(n.mem_capacity for n in env.nodes)),
                        float((X.sum(axis=0) > 0).sum()) / max(1, M),
                    ]
                ],
                dtype=torch.float32,
                device=self.device,
            )
            scores = self.policy.score(svc_embed[:, j], ctx, load_summary, feas_t)
            if self.stochastic_eval:
                probs = F.softmax(scores, dim=-1)
                a = int(torch.multinomial(probs, 1).item())
            else:
                a = int(scores.argmax(dim=-1).item())
            X[a, j] = 1
            svc = env.services[j]
            cpu_used[a] += svc.cpu_demand
            mem_used[a] += svc.mem_demand
        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        scenario_factory,            # callable(seed) -> EpisodeConfig
        train_cfg: TrainConfig,
        log_callback=None,
    ) -> List[dict]:
        opt = torch.optim.Adam(self.policy.parameters(), lr=train_cfg.learning_rate)
        rng = np.random.default_rng(train_cfg.seed)
        history: List[dict] = []
        episodes_done = 0
        while episodes_done < train_cfg.total_episodes:
            transitions: List[Transition] = []
            episode_rewards: List[float] = []
            for _ in range(train_cfg.rollout_episodes):
                seed = int(rng.integers(0, 1_000_000))
                cfg = scenario_factory(seed)
                env = NTNEnvironment(cfg)
                obs = env.reset()
                ep_reward = 0.0
                ep_transitions: List[Transition] = []
                for t in range(cfg.horizon):
                    trans = self._rollout_one_step(env, obs, deterministic=False)
                    ep_transitions.append(trans)
                    ep_reward += trans.reward
                    if trans.done:
                        break
                    env.step()
                    obs = env.observe()
                # Mark final transition as done
                if ep_transitions:
                    ep_transitions[-1].done = True
                transitions.extend(ep_transitions)
                episode_rewards.append(ep_reward)
                episodes_done += 1
                if episodes_done >= train_cfg.total_episodes:
                    break
            # PPO update
            losses = self._ppo_update(transitions, train_cfg, opt)
            mean_ep_reward = float(np.mean(episode_rewards))
            log = {
                "episodes_done": episodes_done,
                "mean_episode_reward": mean_ep_reward,
                **losses,
            }
            history.append(log)
            if log_callback is not None:
                log_callback(log)
        return history

    def _rollout_one_step(
        self,
        env: NTNEnvironment,
        obs: Observation,
        deterministic: bool,
    ) -> Transition:
        N = len(env.nodes)
        M = len(env.services)
        node_t = torch.from_numpy(obs.node_features).float().unsqueeze(0).to(self.device)
        svc_t = torch.from_numpy(obs.service_features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            ctx, svc_embed = self.policy.encode(node_t, svc_t)
            value = self.policy.value(ctx, svc_embed).squeeze(0)

        order = self._service_priority_order(env)
        feas_per_step = np.zeros((M, N), dtype=np.float32)
        load_summaries = np.zeros((M, 3), dtype=np.float32)
        actions = np.full(M, -1, dtype=np.int64)
        log_probs = np.zeros(M, dtype=np.float32)

        cpu_used = np.zeros(N)
        mem_used = np.zeros(N)
        X = np.zeros((N, M), dtype=np.int32)
        for j in order:
            feas = self._capacity_feasible(env, cpu_used, mem_used, j)
            feas_per_step[j] = feas.astype(np.float32)
            load_summaries[j] = [
                cpu_used.sum() / max(1.0, sum(n.cpu_capacity for n in env.nodes)),
                mem_used.sum() / max(1.0, sum(n.mem_capacity for n in env.nodes)),
                float((X.sum(axis=0) > 0).sum()) / max(1, M),
            ]
            if not feas.any():
                # No node can host this service this round: action sentinel -1
                continue
            with torch.no_grad():
                feas_t = torch.from_numpy(feas).bool().unsqueeze(0).to(self.device)
                ls_t = torch.from_numpy(load_summaries[j]).float().unsqueeze(0).to(self.device)
                logits = self.policy.score(svc_embed[:, j], ctx, ls_t, feas_t)
                if deterministic:
                    a = int(logits.argmax(dim=-1).item())
                    log_p = float(F.log_softmax(logits, dim=-1)[0, a].item())
                else:
                    probs = F.softmax(logits, dim=-1)
                    a = int(torch.multinomial(probs, 1).item())
                    log_p = float(torch.log(probs[0, a] + 1e-9).item())
            actions[j] = a
            log_probs[j] = log_p
            X[a, j] = 1
            svc = env.services[j]
            cpu_used[a] += svc.cpu_demand
            mem_used[a] += svc.mem_demand

        info = env.apply_placement(X)
        # Negative cost as reward, normalised so updates stay in a sane range.
        # 1/1000 keeps gradients moderate even when an early-training episode
        # incurs heavy unserved-demand penalties.
        reward = -info.cost_total / 1000.0

        return Transition(
            obs_node=torch.from_numpy(obs.node_features).float(),
            obs_svc=torch.from_numpy(obs.service_features).float(),
            feas_mask_per_step=torch.from_numpy(feas_per_step),
            load_summaries=torch.from_numpy(load_summaries),
            actions=torch.from_numpy(actions),
            log_probs=torch.from_numpy(log_probs),
            value=value,
            reward=reward,
            done=False,
        )

    def _ppo_update(
        self,
        transitions: List[Transition],
        cfg: TrainConfig,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        # Compute returns and advantages with simple per-trajectory GAE,
        # treating each environment step as one transition.
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        values = np.array([float(t.value.item()) for t in transitions], dtype=np.float32)
        dones = np.array([float(t.done) for t in transitions], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for i in reversed(range(len(rewards))):
            next_value = 0.0 if dones[i] else (values[i + 1] if i + 1 < len(values) else 0.0)
            delta = rewards[i] + cfg.gamma * next_value - values[i]
            if dones[i]:
                gae = 0.0
            gae = delta + cfg.gamma * cfg.gae_lambda * gae
            advantages[i] = gae
        returns = advantages + values
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        adv_t = torch.from_numpy(advantages).float().to(self.device)
        ret_t = torch.from_numpy(returns).float().to(self.device)

        idxs = np.arange(len(transitions))
        loss_log = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_batches = 0
        for _ in range(cfg.epochs_per_update):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                pl, vl, ent = self._ppo_minibatch(
                    [transitions[i] for i in mb], adv_t[mb], ret_t[mb], cfg
                )
                optimizer.zero_grad()
                loss = pl + cfg.value_coef * vl - cfg.entropy_coef * ent
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
                optimizer.step()
                loss_log["policy_loss"] += float(pl.item())
                loss_log["value_loss"] += float(vl.item())
                loss_log["entropy"] += float(ent.item())
                n_batches += 1
        for k in loss_log:
            loss_log[k] /= max(1, n_batches)
        return loss_log

    def _ppo_minibatch(
        self,
        transitions: List[Transition],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        cfg: TrainConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Transitions can have different N and M (the training factory
        # randomises these), so we process each transition individually and
        # accumulate the per-decision losses, then average at the end.
        new_log_probs_list = []
        old_log_probs_list = []
        entropy_list = []
        adv_repeats = []
        value_preds = []

        for b, t in enumerate(transitions):
            node_t = t.obs_node.to(self.device).unsqueeze(0)
            svc_t = t.obs_svc.to(self.device).unsqueeze(0)
            ctx, svc_embed = self.policy.encode(node_t, svc_t)
            value = self.policy.value(ctx, svc_embed).squeeze(0)
            value_preds.append(value)

            feas = t.feas_mask_per_step.to(self.device)  # (M, N)
            loads = t.load_summaries.to(self.device)     # (M, 3)
            actions = t.actions.to(self.device)          # (M,)
            old_lp = t.log_probs.to(self.device)         # (M,)
            valid = actions >= 0
            if not valid.any():
                continue
            svc_embed_b = svc_embed[0]   # (M, d)
            ctx_b = ctx[0]               # (N, d)
            q = self.policy.query_mlp(torch.cat([svc_embed_b, loads], dim=-1))  # (M, d)
            scores = (q.unsqueeze(1) * ctx_b.unsqueeze(0)).sum(dim=-1) / (ctx_b.shape[-1] ** 0.5)
            scores = scores.masked_fill(~feas.bool(), float("-inf"))
            log_p_all = F.log_softmax(scores, dim=-1)  # (M, N)
            chosen_lp = log_p_all.gather(1, actions.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            chosen_lp = chosen_lp[valid]
            new_log_probs_list.append(chosen_lp)
            old_log_probs_list.append(old_lp[valid])
            probs = log_p_all.exp()
            ent = -(probs * log_p_all.clamp(min=-1e3)).sum(dim=-1)
            entropy_list.append(ent[valid])
            adv_repeats.extend(
                [float(advantages[b].item())] * int(valid.sum().item())
            )

        if not new_log_probs_list:
            zero = torch.zeros((), device=self.device)
            return zero, zero, zero

        new_lp = torch.cat(new_log_probs_list)
        old_lp = torch.cat(old_log_probs_list)
        entropy = torch.cat(entropy_list).mean()
        adv_t = torch.tensor(adv_repeats, dtype=torch.float32, device=self.device)
        values = torch.stack(value_preds)

        ratio = torch.exp(new_lp - old_lp)
        unclipped = ratio * adv_t
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, returns)
        return policy_loss, value_loss, entropy

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)
