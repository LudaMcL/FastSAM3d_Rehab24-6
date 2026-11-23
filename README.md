SAM 3D Body on REHAB24-6: Exercise- and View-Dependent Agreement With Motion Capture
Evaluation code for the manuscript "Exercise- and View-Dependent Agreement Between SAM 3D Body and Motion Capture for Rehabilitation Exercises" (Y.-L. Liang, L. Y. Deng, X.-Y. Lim, C.-E. Wu; under review at IEEE Transactions on Neural Systems and Rehabilitation Engineering).

The notebook runs the SAM 3D Body foundation model zero-shot (frozen weights, no fine-tuning) on the public REHAB24-6 rehabilitation dataset. It compares predictions from two independently processed camera views with marker-based motion capture for:

joint position;
joint velocity;
repetition-level upper-limb range of motion (ROM).
Inference uses Fast SAM 3D Body, which keeps the architecture and released weights of SAM 3D Body and changes only how inference is run.

Contents
File	Description
eval_fastsam3dbody_rehab24_1.ipynb	Google Colab notebook: setup, inference, metrics, tables, and figures
Main findings
Measure	Result
Valid pose estimates	99.92% of frames
12-joint MPJPE (repetition-aligned)	76.1 mm (95% CI 67.3–85.4)
17-joint MPJPE	97.9 mm
MPJVE (17 joints)	200.9 mm/s
ROM concordance (48 exercise–camera–feature groups)	CCC 0.013 to 0.971
Pooled vs per-exercise (Camera 18, right arm-to-trunk)	pooled CCC 0.914, push-ups only 0.013
The largest ROM over- and underestimation occurred when the arm moved mainly along the camera's line of sight. Validate for each exercise, camera view, and movement feature. Do not rely on pooled accuracy.

Requirements
Google Colab with a GPU. Results in the paper were produced on an NVIDIA A100-SXM4-40GB.
Approved Hugging Face access to facebook/sam-3d-body-dinov3. The weights are gated, and being logged in is not enough while the request is pending.
REHAB24-6 dataset, from the dataset authors' release (see Citation), stored in Google Drive.
Python packages are installed by the notebook (Cell 1), one at a time so that a single failed build does not silently skip the others. These include hydra-core, timm, ultralytics, smplx, opencv-python, pandas, and MoGe (git+https://github.com/microsoft/MoGe.git). detectron2 is not required.

Setup
1. Hugging Face token. In Colab, open Secrets (key icon), add HF_TOKEN, and enable notebook access. Cell 2d logs in and downloads the checkpoint to checkpoints/sam-3d-body-dinov3/ (model.ckpt, model_config.yaml, assets/mhr_model.pt). YOLO11n weights download automatically.

2. Dataset on Google Drive, in this layout:

MyDrive/REHAB/
  videos/Ex1/PM_109-Camera17-30fps.mp4
  videos/Ex1/PM_109-Camera18-30fps-transposed.mp4
  2d_joints/Ex1/PM_109-c17-30fps.npy
  2d_joints/Ex1/PM_109-c18-30fps.npy
  3d_joints/Ex1/PM_109-30fps.npy
  Segmentation.csv
3. Open the notebook in Colab and run the cells from the top. Cell 2a clones Fast SAM 3D Body to /content/fast_sam_3d_body. Cell 2b applies its acceleration settings.

Cell 2b must run before sam_3d_body is imported. If it was already imported in the session, restart the runtime and start again from Cell 2a.

Usage
Step	Cells	What it does
Setup	1 – 3	Install, clone, configure, mount Drive, download weights, build the estimator
Checks on one segment	3b – 12b	Probe 2D keypoint frame and axes, map joints, run one repetition, overlay and metric sanity checks, configuration A/B test (per-frame vs per-video intrinsics; body vs full)
Full dataset	15b → 15w → 15c → 15d	Four parallel workers fill a resumable prediction cache on Drive. Metrics are then computed from the cache
Analysis	16 – 22c	Position and velocity error, ROM agreement (MAE, CCC, Bland–Altman), pooling check, correct vs incorrect execution, paired cross-camera comparison, runtime, qualitative cases
Export and figures	23 – 24c	Tables and figures written to Drive; ROM scatter plots, CCC/MAE heatmaps
Notes:

The prediction cache is keyed by configuration, so predictions from different settings are never mixed. Changing an analysis choice does not re-run inference.
If a Colab session disconnects, re-run Cell 15w. It resumes from the cache.
Run the runtime cells (21, 21b) in a single process after the workers have stopped. Timings taken during parallel inference are not per-frame latency.
Configuration
Setting	Value
Weights	facebook/sam-3d-body-dinov3 (DINOv3 ViT-H+ encoder)
Acceleration	Read directly from Fast SAM 3D Body's run_demo.sh: fewer intermediate decoder predictions, keypoint-prompted refinement pass skipped, MHR correctives disabled, torch.compile for backbone and decoder
Runtime arm	torch.compile, no TensorRT (USE_TRT = False)
Person detector	RT-DETR
Person selection	IoU tracking within each segment; re-initialized to the largest box when IoU < 0.1
Camera intrinsics	MoGe-2, FOV_MODE = "per-video": median of 5 frames per video, then frozen
Hand boxes	HAND_BOX_SOURCE = "body_decoder"
Inference type	full (body + hand decoder)
Segmentation mask	Off (USE_MASK = False)
2D keypoints	Native pred_keypoints_2d, original-frame pixels
Evaluation protocol
Keypoints. 17 of the 70 MHR-70 keypoints are mapped to the 26-joint REHAB24-6 rig, with the hip midpoint as root.
Alignment. Each repetition gets one similarity transform (rotation, uniform scale, translation) that is held fixed across frames, plus hip-midpoint centering.
Position and velocity metrics. MPJPE over 12 limb joints (primary) and all 17 joints (secondary), and MPJVE over 17 joints.
ROM. Elbow and arm-to-trunk ROM per repetition, summarized by MAE and Lin's concordance correlation coefficient (CCC).
Uncertainty. Participant-level cluster bootstrap (4,000 replicates, percentile 95% CIs) and leave-one-participant-out.
Runtime
Measured on an NVIDIA A100-SXM4-40GB, single process:

Measure	Result
Per-frame call (detection + cropping + body-and-hand inference)	mean 239.1 ms (4.18 frames/s); median 215.3 ms; p95 288.6 ms; 1,313 frames from 10 sampled segments
Streaming wall-clock throughput	4.34 frames/s
Not included	One-off per-video intrinsics calibration
This is below the 30 frames/s recording rate, so the setup suits offline analysis of recorded sessions.

Limitations
Relative articulation after reference-assisted alignment, not absolute camera-frame accuracy.
10 participants, scripted exercises, one dataset. Transfer to patient populations is untested.
Per-repetition upper-limb ROM only; lower-limb angles were not analyzed.
The original (non-accelerated) SAM 3D Body implementation was not run on REHAB24-6.
Clinical acceptability is not established.
Citation
If you use this code, please cite the manuscript and the works it builds on:

@article{liang2026sam3dbody_rehab,
  title   = {Exercise- and View-Dependent Agreement Between {SAM} {3D} Body and Motion Capture for Rehabilitation Exercises},
  author  = {Liang, Yuan-Lin and Deng, Lawrence Y. and Lim, Xiang-Yann and Wu, Cheng-En},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  note    = {Under review},
  year    = {2026}
}

@inproceedings{yang2026sam3dbody,
  title     = {{SAM} {3D} Body: Robust Full-Body Human Mesh Recovery},
  author    = {Yang, Xitong and others},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}

@inproceedings{yang2026fastsam3dbody,
  title     = {Fast {SAM} {3D} Body: Accelerating {SAM} {3D} Body for Real-Time Full-Body Human Mesh Recovery},
  author    = {Yang, Timing and He, Sicheng and Jing, Hongyi and Yang, Jiawei and Liu, Zhijian and Zou, Chuhang and Wang, Yue},
  booktitle = {European Conference on Computer Vision (ECCV)},
  series    = {Lecture Notes in Computer Science},
  volume    = {17012},
  pages     = {262--279},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-3-032-37044-0_15}
}

@inproceedings{cernek2024rehab246,
  title     = {{REHAB24-6}: Physical Therapy Dataset for Analyzing Pose Estimation Methods},
  author    = {{\v{C}}ernek, A. and Sedmidubsk{\'y}, J. and Bud{\'i}kov{\'a}, P.},
  booktitle = {Similarity Search and Applications (SISAP)},
  series    = {Lecture Notes in Computer Science},
  volume    = {15268},
  pages     = {18--33},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-75823-2_2}
}
Acknowledgments and licenses
This work uses SAM 3D Body (Meta), Fast SAM 3D Body, MoGe, Ultralytics YOLO, and the REHAB24-6 dataset. Each is subject to its own license and terms of use; check them before redistributing weights or data.
