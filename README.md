# A Multi-Stage XAI Analysis of MAC-VO

**BBM416/AIN431 – Fundamentals of Computer Vision, Final Project**
Hacettepe University, Department of Artificial Intelligence Engineering

M. Beşir Acar · Ahmet Emre Gökpınar

📺 [Project video](https://youtu.be/stk2wyLCTUY)

---

## Overview

[MAC-VO](https://arxiv.org/abs/2409.09479) is a learning-based stereo visual odometry (VO) system whose key idea is a *metrics-aware covariance*: its FlowFormerCov frontend predicts dense optical flow together with a per-pixel 2D uncertainty, which is then propagated through keypoint selection, outlier rejection, and pose-graph optimization. That uncertainty is a load-bearing part of the pipeline — but the network producing it is a black box.

This project extends MAC-VO with **nine post-hoc explainable-AI (XAI) analyses**, applied across two benchmarks (**TartanAir** and **EuRoC MH_01_easy**) and all three pipeline stages (frontend, keypoint selector, trajectory back-end), to answer three questions:

1. What image evidence does the predicted covariance actually respond to?
2. Does covariance-driven keypoint selection have systematic blind spots?
3. Which conditions drive pose error?

## Key Findings

- **The learned covariance is semantically grounded.** GradCAM saliency, easy/hard activation contrast, and a Harris-corner anti-correlation (`r = -0.305, p ≈ 1e-4`) all show the covariance head attending to texture and matching ambiguity, and it stays temporally stationary rather than drifting.
- **Keypoint selection has a strong, reproducible center bias.** Accepted keypoints sit further from the image border than rejected ones on TartanAir (~20 px gap), and rejection rates on EuRoC reach up to **0.73** at the borders. This is a deliberate robustness mechanism, but also a quantified spatial blind spot — and it appears with two completely different camera setups.
- **Failure drivers are dataset-dependent.** Motion blur is the dominant predictor of pose error on real EuRoC imagery (Spearman `r = 0.422, p < 0.001`), but is uninformative on synthetic TartanAir data — motion blur simply isn't rendered there.
- **A LIME-style trajectory perturbation** shows global trajectory accuracy hinges on a small set of currently well-estimated segments, not on uniformly distributed risk.

## Methodology

Nine XAI methods were organized into two studies:

| Study | Stage | Method | Question Answered |
|---|---|---|---|
| TartanAir | Whole system | GradCAM on covariance head | Where does the network look when predicting uncertainty? |
| TartanAir | Whole system | Image-statistics failure attribution | Do blur, brightness, or contrast predict pose error? |
| TartanAir | Whole system | Uncertainty–error calibration | Does higher uncertainty imply higher actual error? |
| TartanAir | Whole system | Temporal covariance audit | Is uncertainty stationary, drifting, or scene-driven? |
| TartanAir | Whole system | Keypoint density & edge distance | Where are keypoints kept or discarded? |
| EuRoC | Frontend | Easy/hard activation contrast | Which conditions raise uncertainty? |
| EuRoC | Keypoint selector | Spatial rejection-bias map | Does selection have systematic blind spots? |
| EuRoC | Back-end | Per-axis drift decomposition | Which axis drifts? |
| EuRoC | Back-end | LIME-style trajectory perturbation | Which trajectory segments are critical? |
| EuRoC | Back-end | Image-feature attribution | Which image features predict failure? |

Methods span **gradient attribution** (GradCAM), **causal perturbation** (LIME-style segment noising), and **statistical/correlation-based auditing** (Pearson/Spearman correlation, Kolmogorov–Smirnov tests, distance-transform edge analysis, per-axis RMSE decomposition).

> **Note on error metrics:** per-frame error values use centroid-only trajectory alignment (no SE(3)/Umeyama alignment), which inflates absolute error relative to the properly aligned benchmark numbers below. Correlation-based conclusions are unaffected, but absolute magnitudes across the two error definitions are not directly comparable.

## Experimental Setup

- **Hardware:** Single NVIDIA Tesla T4 (16 GB)
- **Model:** Official MAC-VO repository, released pretrained weights (`MACVO_FrontendCov.pth`), no fine-tuning, fp32 frontend with a 12-block decoder
- **Datasets:**
  - **TartanAir** `abf001/P001_select` — synthetic abandoned-factory sequence, 1,274 stereo pairs, exact ground truth
  - **EuRoC** `MH_01_easy` — real machine-hall MAV sequence, 3,682 stereo frames (3,616 estimated poses after IMU/camera alignment)

**Baseline trajectory accuracy (MAC-VO Performant):**

| Metric | TartanAir abf001 (1,274 frames) | EuRoC MH_01_easy (3,616 poses) |
|---|---|---|
| ATE — μ / σ / RMSE (m) | 1.4554 / 1.1586 / 1.8602 | 0.1766 / 0.0832 / 0.1952 |
| RTE — μ / RMSE | 0.0044 / 0.0104 | 0.0014 / 0.0017 |
| ROE — μ / RMSE | 0.0308 / 0.0744 | 0.0153 / 0.0178 |
| RPE — μ / RMSE | 0.0046 / 0.0106 | 0.0014 / 0.0018 |

## Repository Contents

- Analysis notebooks/scripts for all nine XAI methods across both datasets
- Generated figures (GradCAM overlays, calibration curves, temporal covariance audits, spatial rejection-bias maps, per-axis drift plots, LIME sensitivity plots, feature-correlation matrices)
- Final report (this project's write-up, ICLR-style)
- Presentation materials

## Discussion & Limitations

- Applying GradCAM required temporarily disabling the CUDA-graph-wrapped frontend to allow eager-mode backward hooks.
- GPU memory constraints (16 GB T4) required `torch.cuda.empty_cache()` after every frame and strict `inference_mode()` usage across repeated frontend passes (150 runs for the temporal audit, 85 for the keypoint density map).
- The step-size-based calibration proxy produced a flat calibration curve (`r = 0.194, p = 0.41`), indicating the proxy itself is weak — not necessarily the underlying covariance. A KS test on the same data (`D = 0.128, p = 6.1e-3`) confirms the covariance does separate error distributions even where the proxy fails to show a monotone relationship.
- Correlation-based methods (7 of the 9 analyses) are observational, not causal — e.g., the blur–error correlation cannot rule out camera motion as a common cause.

**Suggested future work:** replace the step-size calibration proxy with the network's own per-pixel covariance evaluated against Umeyama-aligned error; extend to more sequences; fold the blur and covariance signals into a real-time deployment quality check for UAV search-and-rescue use cases.

## References

- Qiu et al., *MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry*, 2025. [arXiv:2409.09479](https://arxiv.org/abs/2409.09479)
- Huang et al., *FlowFormer: A Transformer Architecture for Optical Flow*, 2022. [arXiv:2203.16194](https://arxiv.org/abs/2203.16194)
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, IJCV 2019.
- Ribeiro et al., *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*, 2016. [arXiv:1602.04938](https://arxiv.org/abs/1602.04938)
- Wang et al., *TartanAir: A Dataset to Push the Limits of Visual SLAM*, 2020. [arXiv:2003.14338](https://arxiv.org/abs/2003.14338)
- Burri et al., *The EuRoC Micro Aerial Vehicle Datasets*, IJRR 2016.

## Authors

- **M. Beşir Acar** — Department of Artificial Intelligence Engineering, Hacettepe University — `b222075003@cs.hacettepe.edu.tr`
- **Ahmet Emre Gökpınar** — Department of Computer Engineering, Hacettepe University — `b2200356095@cs.hacettepe.edu.tr`
