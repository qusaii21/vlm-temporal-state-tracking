# Dataset Information: Video-MME Logical (Card-Relocation Subset)

This benchmark evaluates temporal state tracking using the **card-relocation** subset of the `video-mme-logical` benchmark ([Hugging Face: `marcuskwan/video-mme-logical`](https://huggingface.co/datasets/marcuskwan/video-mme-logical)).

## Task Description
- **Setup**: Four cards sit at four fixed positions (`A`, `B`, `C`, `D`).
- **Events**: One card is designated as the target card, and the video displays six sequential rounds of positional swaps.
- **Goal**: Predict the target card's position after each intermediate round (1 to 6) and at the final state.
- **Evaluation Subset**: 50 videos (40 training videos, 10 held-out evaluation videos per fold in 5-fold cross-validation).

## Files in this Directory
- `ground_truth_manifest_50.json`: Complete ground truth metadata for all 50 videos, containing video filenames, target card IDs, initial positions, and per-round position traces.
- `split_summary_seed20260917.json`: Fixed split definitions identifying the 10 held-out evaluation videos for Fold 0.

## Downloading Video Files
To download the underlying video clips directly from Hugging Face:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="marcuskwan/video-mme-logical",
    repo_type="dataset",
    local_dir="data/video-mme-logical",
    allow_patterns=["videos/card_relocation_s_easy/*"]
)
```

## Critical Dataset Audit Finding: The Starting-Position Shortcut
Across all 50 videos in this subset, the dataset exhibits a systematic property:
$$\text{target\_card} == \text{initial\_position}$$
Because the evaluation prompt specifies the target card to track (e.g., *"The target card is the card labeled A"*), the initial state is trivially revealed in the prompt text.

Furthermore, across the 300 round evaluations (50 videos $\times$ 6 rounds):
- In **106 rounds (35.3%)**, the target card is located at its initial starting position.
- In **194 rounds (64.7%)**, the target card has moved away from its start.

A trivial **stay-put baseline** (a policy that ignores the video and simply predicts the initial position for every round) achieves **35.0% per-round accuracy**, matching the trajectory-supervised model ($B$).
