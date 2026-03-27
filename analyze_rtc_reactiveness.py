"""
RTC Reactiveness Analysis
=========================
Compares expert vs inference action data across different alpha values to analyze:
1. Which action chunks deviate most from expert data
2. Acceleration and high-frequency ratio of action trajectories
3. How RTC smoothing impacts reactiveness and task success

Hypothesis: RTC over-smooths actions, blocking high-reactiveness needed for certain tasks.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ── Config ──────────────────────────────────────────────────────────────────
EXPERT_DIR = '/home/seunghoon/real-time-chunking-kinetix/logs-expert/data/'
INF_DIR = '/home/seunghoon/real-time-chunking-kinetix/matched_inference/'
OUT_DIR = '/home/seunghoon/real-time-chunking-kinetix/rtc_analysis/'
CHUNK_SIZE = 3
ALPHAS = ['alpha_1.0', 'alpha_3.0', 'alpha_5.0', 'hard_masking']
ALPHA_LABELS = {'alpha_1.0': 'α=1.0', 'alpha_3.0': 'α=3.0',
                'alpha_5.0': 'α=5.0', 'hard_masking': 'No RTC'}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'per_task'), exist_ok=True)

TASKS = [f.replace('.npz', '') for f in sorted(os.listdir(EXPERT_DIR)) if f.endswith('.npz')]


# ── Helpers ─────────────────────────────────────────────────────────────────
def unfold_chunks(action_chunks, length):
    """Unfold (T_chunks, chunk_size, act_dim) -> (T_chunks * chunk_size, act_dim),
    truncated to actual episode length * chunk_size."""
    T = min(length, action_chunks.shape[0])
    return action_chunks[:T].reshape(T * CHUNK_SIZE, -1)


def compute_acceleration(actions):
    """Compute acceleration (2nd derivative) of action trajectory."""
    if len(actions) < 3:
        return np.zeros((0, actions.shape[1]))
    velocity = np.diff(actions, axis=0)
    accel = np.diff(velocity, axis=0)
    return accel


def compute_high_freq_ratio(actions, cutoff_fraction=0.3):
    """Compute ratio of high-frequency energy to total energy via FFT.
    cutoff_fraction: fraction of spectrum considered 'high frequency'."""
    if len(actions) < 4:
        return 0.0
    # Apply FFT per action dimension
    ratios = []
    for dim in range(actions.shape[1]):
        signal = actions[:, dim]
        signal = signal - signal.mean()  # remove DC
        fft_vals = np.fft.rfft(signal)
        power = np.abs(fft_vals) ** 2
        if power.sum() < 1e-10:
            continue
        n_freqs = len(power)
        cutoff = int(n_freqs * (1 - cutoff_fraction))
        high_freq_power = power[cutoff:].sum()
        total_power = power[1:].sum()  # exclude DC
        if total_power > 1e-10:
            ratios.append(high_freq_power / total_power)
    return np.mean(ratios) if ratios else 0.0


def compute_jerk(actions):
    """Compute jerk (3rd derivative) - measures abruptness of motion changes."""
    if len(actions) < 4:
        return np.zeros((0, actions.shape[1]))
    return np.diff(actions, n=3, axis=0)


def per_chunk_metrics(actions, chunk_size=CHUNK_SIZE):
    """Compute per-chunk acceleration magnitude and high-freq content.
    Returns metrics for each chunk boundary and within-chunk stats."""
    n_chunks = len(actions) // chunk_size
    if n_chunks < 2:
        return {}, {}

    # Acceleration at chunk boundaries (between last action of chunk i and first of chunk i+1)
    boundary_accels = []
    within_accels = []

    for c in range(n_chunks):
        start = c * chunk_size
        end = start + chunk_size
        chunk_actions = actions[start:end]

        # Within-chunk acceleration
        if len(chunk_actions) >= 3:
            acc = compute_acceleration(chunk_actions)
            within_accels.append(np.linalg.norm(acc, axis=1).mean())

        # Boundary acceleration (between this chunk's end and next chunk's start)
        if c < n_chunks - 1:
            boundary_region = actions[end - 1:end + 2]  # last of current, first two of next
            if len(boundary_region) >= 3:
                acc = compute_acceleration(boundary_region)
                boundary_accels.append(np.linalg.norm(acc, axis=1).mean())

    return {
        'boundary_accel': np.array(boundary_accels),
        'within_accel': np.array(within_accels),
    }, {
        'mean_boundary_accel': np.mean(boundary_accels) if boundary_accels else 0,
        'mean_within_accel': np.mean(within_accels) if within_accels else 0,
    }


# ── Main Analysis ───────────────────────────────────────────────────────────
results = {}
all_chunk_deviations = {}  # task -> alpha -> per-chunk deviation from expert

print("=" * 100)
print("RTC REACTIVENESS ANALYSIS")
print("=" * 100)

for task in TASKS:
    print(f"\n{'─' * 80}")
    print(f"Task: {task}")
    print(f"{'─' * 80}")

    # Load expert data
    expert = np.load(os.path.join(EXPERT_DIR, f'{task}.npz'), allow_pickle=True)
    expert_actions = expert['action']      # (N, T_expert, act_dim)
    expert_solved = expert['solved']       # (N,)
    expert_lengths = expert['length']      # (N,)
    N = expert_actions.shape[0]

    # Expert metrics (aggregate)
    expert_accels = []
    expert_hf_ratios = []
    expert_jerks = []
    for i in range(N):
        T = expert_lengths[i]
        acts = expert_actions[i, :T]
        acc = compute_acceleration(acts)
        if len(acc) > 0:
            expert_accels.append(np.linalg.norm(acc, axis=1).mean())
        expert_hf_ratios.append(compute_high_freq_ratio(acts))
        jrk = compute_jerk(acts)
        if len(jrk) > 0:
            expert_jerks.append(np.linalg.norm(jrk, axis=1).mean())

    task_result = {
        'expert_solved_rate': float(expert_solved.mean()),
        'expert_mean_accel': float(np.mean(expert_accels)),
        'expert_mean_hf_ratio': float(np.mean(expert_hf_ratios)),
        'expert_mean_jerk': float(np.mean(expert_jerks)) if expert_jerks else 0,
        'alphas': {},
    }

    print(f"  Expert: solved={expert_solved.mean():.3f}, "
          f"accel={np.mean(expert_accels):.4f}, "
          f"HF_ratio={np.mean(expert_hf_ratios):.4f}, "
          f"jerk={np.mean(expert_jerks):.4f}")

    all_chunk_deviations[task] = {}

    for alpha in ALPHAS:
        inf_file = os.path.join(INF_DIR, alpha, f'worlds_l_{task}.npz')
        if not os.path.exists(inf_file):
            print(f"  {ALPHA_LABELS[alpha]}: MISSING")
            continue

        inf = np.load(inf_file, allow_pickle=True)
        inf_actions = inf['action']    # (N, T, act_dim)
        inf_solved = inf['solved']     # (N,)
        inf_lengths = inf['length']    # (N,)

        solved_rate = inf_solved.mean()

        inf_accels = []
        inf_hf_ratios = []
        inf_jerks = []
        inf_boundary_accels = []
        inf_within_accels = []
        chunk_deviations_per_ep = []  # per-chunk deviation from expert

        for i in range(N):
            T_inf = inf_lengths[i]
            acts = inf_actions[i, :T_inf]

            # Full trajectory metrics
            acc = compute_acceleration(acts)
            if len(acc) > 0:
                inf_accels.append(np.linalg.norm(acc, axis=1).mean())
            inf_hf_ratios.append(compute_high_freq_ratio(acts))
            jrk = compute_jerk(acts)
            if len(jrk) > 0:
                inf_jerks.append(np.linalg.norm(jrk, axis=1).mean())

            # Per-chunk boundary vs within acceleration
            chunk_metrics, chunk_summary = per_chunk_metrics(acts)
            if chunk_metrics:
                inf_boundary_accels.append(chunk_summary['mean_boundary_accel'])
                inf_within_accels.append(chunk_summary['mean_within_accel'])

            # Per-chunk deviation from expert
            T_expert = expert_lengths[i]
            expert_acts = expert_actions[i, :T_expert]
            T_compare = min(len(acts), len(expert_acts))
            if T_compare > CHUNK_SIZE:
                n_compare_chunks = T_compare // CHUNK_SIZE
                per_chunk_dev = []
                for c in range(n_compare_chunks):
                    s, e = c * CHUNK_SIZE, (c + 1) * CHUNK_SIZE
                    dev = np.linalg.norm(acts[s:e] - expert_acts[s:e], axis=1).mean()
                    per_chunk_dev.append(dev)
                chunk_deviations_per_ep.append(per_chunk_dev)

        # Aggregate chunk deviations across episodes (pad to max length)
        if chunk_deviations_per_ep:
            max_chunks = max(len(d) for d in chunk_deviations_per_ep)
            padded = np.full((len(chunk_deviations_per_ep), max_chunks), np.nan)
            for j, d in enumerate(chunk_deviations_per_ep):
                padded[j, :len(d)] = d
            mean_chunk_dev = np.nanmean(padded, axis=0)
            all_chunk_deviations[task][alpha] = mean_chunk_dev
        else:
            mean_chunk_dev = np.array([])

        alpha_result = {
            'solved_rate': float(solved_rate),
            'mean_accel': float(np.mean(inf_accels)) if inf_accels else 0,
            'mean_hf_ratio': float(np.mean(inf_hf_ratios)),
            'mean_jerk': float(np.mean(inf_jerks)) if inf_jerks else 0,
            'mean_boundary_accel': float(np.mean(inf_boundary_accels)) if inf_boundary_accels else 0,
            'mean_within_accel': float(np.mean(inf_within_accels)) if inf_within_accels else 0,
            'accel_ratio_vs_expert': float(np.mean(inf_accels) / np.mean(expert_accels)) if (inf_accels and np.mean(expert_accels) > 1e-10) else 0,
            'hf_ratio_vs_expert': float(np.mean(inf_hf_ratios) / np.mean(expert_hf_ratios)) if np.mean(expert_hf_ratios) > 1e-10 else 0,
        }
        task_result['alphas'][alpha] = alpha_result

        print(f"  {ALPHA_LABELS[alpha]:>10}: solved={solved_rate:.3f}, "
              f"accel={np.mean(inf_accels):.4f} ({alpha_result['accel_ratio_vs_expert']:.2f}x expert), "
              f"HF={np.mean(inf_hf_ratios):.4f} ({alpha_result['hf_ratio_vs_expert']:.2f}x expert), "
              f"jerk={np.mean(inf_jerks):.4f}, "
              f"boundary/within_accel={np.mean(inf_boundary_accels):.4f}/{np.mean(inf_within_accels):.4f}")

        # Find chunk indices with biggest deviation
        if len(mean_chunk_dev) > 0:
            top_k = min(5, len(mean_chunk_dev))
            top_chunks = np.argsort(mean_chunk_dev)[-top_k:][::-1]
            print(f"            Top deviating chunks: {list(top_chunks)} "
                  f"(devs: {[f'{mean_chunk_dev[c]:.4f}' for c in top_chunks]})")

    results[task] = task_result


# ── Summary Analysis ────────────────────────────────────────────────────────
print("\n\n" + "=" * 100)
print("SUMMARY: REACTIVENESS IMPACT ON TASK SUCCESS")
print("=" * 100)

# Compute per-task "reactiveness gap" = how much RTC reduces HF ratio / acceleration
print(f"\n{'Task':<25} {'Expert SR':>10} {'α=1 SR':>8} {'α=1 HF↓':>10} {'α=1 Acc↓':>10} "
      f"{'NoRTC SR':>10} {'NoRTC HF↓':>10} {'NoRTC Acc↓':>10} {'SR Gap':>8}")

task_reactiveness_data = []
for task in TASKS:
    r = results[task]
    a1 = r['alphas'].get('alpha_1.0', {})
    hm = r['alphas'].get('hard_masking', {})

    hf_gap_a1 = 1.0 - a1.get('hf_ratio_vs_expert', 0)
    acc_gap_a1 = 1.0 - a1.get('accel_ratio_vs_expert', 0)
    hf_gap_hm = 1.0 - hm.get('hf_ratio_vs_expert', 0)
    acc_gap_hm = 1.0 - hm.get('accel_ratio_vs_expert', 0)
    sr_gap_a1 = r['expert_solved_rate'] - a1.get('solved_rate', 0)
    sr_gap_hm = r['expert_solved_rate'] - hm.get('solved_rate', 0)

    print(f"{task:<25} {r['expert_solved_rate']:>10.3f} "
          f"{a1.get('solved_rate', 0):>8.3f} "
          f"{hf_gap_a1:>+10.4f} {acc_gap_a1:>+10.4f} "
          f"{hm.get('solved_rate', 0):>10.3f} "
          f"{hf_gap_hm:>+10.4f} {acc_gap_hm:>+10.4f} "
          f"{sr_gap_a1:>+8.3f}")

    task_reactiveness_data.append({
        'task': task,
        'expert_sr': r['expert_solved_rate'],
        'alpha1_sr': a1.get('solved_rate', 0),
        'hard_masking_sr': hm.get('solved_rate', 0),
        'sr_drop_alpha1': sr_gap_a1,
        'hf_reduction_alpha1': hf_gap_a1,
        'accel_reduction_alpha1': acc_gap_a1,
        'expert_hf': r['expert_mean_hf_ratio'],
        'expert_accel': r['expert_mean_accel'],
    })


# ── Correlation Analysis ───────────────────────────────────────────────────
print("\n\n" + "=" * 100)
print("CORRELATION: Expert Reactiveness vs Success Rate Drop under RTC")
print("=" * 100)

sr_drops = [d['sr_drop_alpha1'] for d in task_reactiveness_data]
expert_hfs = [d['expert_hf'] for d in task_reactiveness_data]
expert_accels = [d['expert_accel'] for d in task_reactiveness_data]
hf_reductions = [d['hf_reduction_alpha1'] for d in task_reactiveness_data]
accel_reductions = [d['accel_reduction_alpha1'] for d in task_reactiveness_data]

corr_hf_sr = np.corrcoef(expert_hfs, sr_drops)[0, 1]
corr_accel_sr = np.corrcoef(expert_accels, sr_drops)[0, 1]
corr_hf_red_sr = np.corrcoef(hf_reductions, sr_drops)[0, 1]
corr_accel_red_sr = np.corrcoef(accel_reductions, sr_drops)[0, 1]

print(f"  Corr(expert_HF_ratio, SR_drop):          {corr_hf_sr:+.4f}")
print(f"  Corr(expert_accel, SR_drop):              {corr_accel_sr:+.4f}")
print(f"  Corr(HF_reduction_by_RTC, SR_drop):       {corr_hf_red_sr:+.4f}")
print(f"  Corr(accel_reduction_by_RTC, SR_drop):     {corr_accel_red_sr:+.4f}")
print("\n  Positive correlation = tasks with higher expert reactiveness suffer more from RTC")


# ── Average metrics across all tasks ────────────────────────────────────────
print("\n\n" + "=" * 100)
print("AVERAGE METRICS ACROSS ALL TASKS")
print("=" * 100)

print(f"\n{'Metric':<30} {'Expert':>10}", end='')
for a in ALPHAS:
    print(f" {ALPHA_LABELS[a]:>12}", end='')
print()

for metric_name, metric_key in [('Acceleration', 'mean_accel'),
                                  ('HF Ratio', 'mean_hf_ratio'),
                                  ('Jerk', 'mean_jerk'),
                                  ('Boundary Accel', 'mean_boundary_accel'),
                                  ('Within-chunk Accel', 'mean_within_accel'),
                                  ('Solved Rate', 'solved_rate')]:
    expert_key = f'expert_{metric_key}' if metric_key not in ['mean_boundary_accel', 'mean_within_accel', 'solved_rate'] else None

    if expert_key and expert_key.replace('expert_', 'expert_mean_') in results[TASKS[0]]:
        expert_key = expert_key.replace('expert_', 'expert_mean_')

    if metric_key == 'solved_rate':
        expert_val = np.mean([results[t]['expert_solved_rate'] for t in TASKS])
    elif f'expert_mean_{metric_key.replace("mean_", "")}' in results[TASKS[0]]:
        expert_val = np.mean([results[t][f'expert_mean_{metric_key.replace("mean_", "")}'] for t in TASKS])
    else:
        expert_val = None

    row = f"{metric_name:<30}"
    if expert_val is not None:
        row += f" {expert_val:>10.4f}"
    else:
        row += f" {'N/A':>10}"

    for a in ALPHAS:
        vals = [results[t]['alphas'][a][metric_key] for t in TASKS if a in results[t]['alphas']]
        if vals:
            row += f" {np.mean(vals):>12.4f}"
        else:
            row += f" {'N/A':>12}"
    print(row)


# ── Plotting ────────────────────────────────────────────────────────────────
print("\n\nGenerating plots...")

# Plot 1: Per-task success rate comparison
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(TASKS))
width = 0.15
bars = [ax.bar(x - 2*width, [results[t]['expert_solved_rate'] for t in TASKS],
               width, label='Expert', color='black', alpha=0.8)]
for i, alpha in enumerate(ALPHAS):
    vals = [results[t]['alphas'].get(alpha, {}).get('solved_rate', 0) for t in TASKS]
    bars.append(ax.bar(x + (i-1)*width, vals, width, label=ALPHA_LABELS[alpha]))
ax.set_xticks(x)
ax.set_xticklabels(TASKS, rotation=45, ha='right')
ax.set_ylabel('Success Rate')
ax.set_title('Task Success Rate: Expert vs RTC Inference (Different α)')
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_success_rate_comparison.png'), dpi=150)
plt.close()

# Plot 2: Acceleration ratio (inference/expert) per task
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax_idx, (metric, metric_key, expert_metric) in enumerate([
    ('Acceleration', 'mean_accel', 'expert_mean_accel'),
    ('HF Ratio', 'mean_hf_ratio', 'expert_mean_hf_ratio'),
]):
    ax = axes[ax_idx]
    for i, alpha in enumerate(ALPHAS):
        ratios = []
        for t in TASKS:
            if alpha in results[t]['alphas']:
                inf_val = results[t]['alphas'][alpha][metric_key]
                exp_val = results[t][expert_metric]
                ratios.append(inf_val / exp_val if exp_val > 1e-10 else 0)
            else:
                ratios.append(0)
        ax.bar(x + (i - 1.5) * width, ratios, width, label=ALPHA_LABELS[alpha])
    ax.axhline(y=1.0, color='black', linestyle='--', label='Expert baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=45, ha='right')
    ax.set_ylabel(f'{metric} (ratio vs Expert)')
    ax.set_title(f'{metric}: Inference / Expert Ratio')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_reactiveness_ratio.png'), dpi=150)
plt.close()

# Plot 3: Scatter - Expert reactiveness vs success rate drop
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax_idx, (metric_vals, label) in enumerate([
    (expert_hfs, 'Expert HF Ratio'),
    (expert_accels, 'Expert Acceleration'),
]):
    ax = axes[ax_idx]
    ax.scatter(metric_vals, sr_drops, s=80, c='steelblue', edgecolors='black')
    for i, task in enumerate(TASKS):
        ax.annotate(task, (metric_vals[i], sr_drops[i]), fontsize=7,
                    xytext=(5, 5), textcoords='offset points')
    corr = np.corrcoef(metric_vals, sr_drops)[0, 1]
    ax.set_xlabel(label)
    ax.set_ylabel('Success Rate Drop (Expert - α=1.0)')
    ax.set_title(f'{label} vs SR Drop (r={corr:.3f})')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_reactiveness_vs_sr_drop.png'), dpi=150)
plt.close()

# Plot 4: Per-chunk deviation from expert (averaged across episodes)
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for t_idx, task in enumerate(TASKS):
    ax = axes[t_idx // 4, t_idx % 4]
    for alpha in ALPHAS:
        if alpha in all_chunk_deviations[task]:
            devs = all_chunk_deviations[task][alpha]
            ax.plot(devs, label=ALPHA_LABELS[alpha], alpha=0.8)
    ax.set_title(task, fontsize=9)
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('L2 Deviation')
    if t_idx == 0:
        ax.legend(fontsize=7)
plt.suptitle('Per-Chunk Action Deviation from Expert', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_per_chunk_deviation.png'), dpi=150)
plt.close()

# Plot 5: Boundary vs Within-chunk acceleration
fig, ax = plt.subplots(figsize=(14, 6))
for i, alpha in enumerate(ALPHAS):
    boundary = [results[t]['alphas'].get(alpha, {}).get('mean_boundary_accel', 0) for t in TASKS]
    within = [results[t]['alphas'].get(alpha, {}).get('mean_within_accel', 0) for t in TASKS]
    ratio = [b / w if w > 1e-10 else 0 for b, w in zip(boundary, within)]
    ax.bar(x + (i - 1.5) * width, ratio, width, label=ALPHA_LABELS[alpha])
ax.set_xticks(x)
ax.set_xticklabels(TASKS, rotation=45, ha='right')
ax.set_ylabel('Boundary / Within-Chunk Acceleration Ratio')
ax.set_title('Chunk Boundary Acceleration Ratio (Higher = More Discontinuity at Boundaries)')
ax.legend()
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_boundary_vs_within_accel.png'), dpi=150)
plt.close()

# Plot 6: Heatmap - alpha effect on HF ratio reduction per task
fig, ax = plt.subplots(figsize=(10, 8))
hf_matrix = np.zeros((len(TASKS), len(ALPHAS)))
for t_idx, task in enumerate(TASKS):
    for a_idx, alpha in enumerate(ALPHAS):
        if alpha in results[task]['alphas']:
            hf_matrix[t_idx, a_idx] = results[task]['alphas'][alpha].get('hf_ratio_vs_expert', 0)
im = ax.imshow(hf_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
ax.set_xticks(range(len(ALPHAS)))
ax.set_xticklabels([ALPHA_LABELS[a] for a in ALPHAS])
ax.set_yticks(range(len(TASKS)))
ax.set_yticklabels(TASKS)
for t_idx in range(len(TASKS)):
    for a_idx in range(len(ALPHAS)):
        ax.text(a_idx, t_idx, f'{hf_matrix[t_idx, a_idx]:.2f}',
                ha='center', va='center', fontsize=8)
plt.colorbar(im, label='HF Ratio (inference / expert)')
ax.set_title('High-Frequency Ratio: Inference vs Expert\n(< 1 = RTC suppresses reactiveness)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06_hf_ratio_heatmap.png'), dpi=150)
plt.close()

# Plot 7: Identify which chunk index makes biggest difference across tasks
fig, ax = plt.subplots(figsize=(12, 6))
# Normalize chunk deviations to relative position (0-1) and aggregate
n_bins = 20
binned_devs = {alpha: np.zeros(n_bins) for alpha in ALPHAS}
binned_counts = {alpha: np.zeros(n_bins) for alpha in ALPHAS}

for task in TASKS:
    for alpha in ALPHAS:
        if alpha in all_chunk_deviations[task]:
            devs = all_chunk_deviations[task][alpha]
            n = len(devs)
            for c_idx, dev in enumerate(devs):
                bin_idx = min(int(c_idx / n * n_bins), n_bins - 1)
                binned_devs[alpha][bin_idx] += dev
                binned_counts[alpha][bin_idx] += 1

for alpha in ALPHAS:
    mask = binned_counts[alpha] > 0
    avg = np.zeros(n_bins)
    avg[mask] = binned_devs[alpha][mask] / binned_counts[alpha][mask]
    ax.plot(np.linspace(0, 1, n_bins), avg, label=ALPHA_LABELS[alpha], marker='o', markersize=4)

ax.set_xlabel('Relative Position in Episode (0=start, 1=end)')
ax.set_ylabel('Mean L2 Deviation from Expert')
ax.set_title('Where in the Episode Does RTC Deviate Most from Expert?')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_deviation_by_episode_position.png'), dpi=150)
plt.close()

# Save results to JSON
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nAll plots and results saved to: {OUT_DIR}")
print("Done!")
