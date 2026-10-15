# Secondary confirmations

All required DistilBERT primary and ablation runs precede these confirmations. Models and failures are separate from primary hypothesis testing. No favorable seed is selected.

| family | method | seed | status | failure_reason |
|---|---|---|---|---|
| deberta | baseline | 13 | FAILED | Return code 1; retained log: ./outputs\secondary_logs\deberta_baseline_13_20261013T075907Z.log |
| deberta | baseline | 42 | PENDING | Earlier architecture attempt failed; optional confirmation stopped without altering hyperparameters. |
| deberta | baseline | 123 | PENDING | Earlier architecture attempt failed; optional confirmation stopped without altering hyperparameters. |
| deberta | provguard_1000 | 13 | PENDING | Earlier architecture attempt failed; optional confirmation stopped without altering hyperparameters. |
| deberta | provguard_1000 | 42 | PENDING | Earlier architecture attempt failed; optional confirmation stopped without altering hyperparameters. |
| deberta | provguard_1000 | 123 | PENDING | Earlier architecture attempt failed; optional confirmation stopped without altering hyperparameters. |
| source | provguard_dolly_250 | 13 | COMPLETED | unavailable |
| source | provguard_dolly_250 | 42 | COMPLETED | unavailable |
| source | provguard_dolly_250 | 123 | COMPLETED | unavailable |
| source | provguard_oasst1_250 | 13 | COMPLETED | unavailable |
| source | provguard_oasst1_250 | 42 | COMPLETED | unavailable |
| source | provguard_oasst1_250 | 123 | COMPLETED | unavailable |

| family | method | seed | augmentation_size | iid_f1 | iid_recall | hard_benign_fpr | external_hb_fpr | obfuscated_recall | config_sha256 | dolly_n | dolly_fpr | oasst1_n | oasst1_fpr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| source | provguard_dolly_250 | 13 | 250 | 0.9828 | 0.9908 | 0.3886 | 0.1635 | 0.9901 | 98c0f0f3d7f201d57c0717083c5d01696380023bf7dc1eab6606fb2c98a1b9fe | 68 | 0.0000 | 140 | 0.2429 |
| source | provguard_dolly_250 | 42 | 250 | 0.9818 | 0.9884 | 0.2203 | 0.1202 | 0.9802 | 87c5014251f74c79f3939fc80f0205d56635dc19ca5e9113aeabe49f17480813 | 68 | 0.0000 | 140 | 0.1786 |
| source | provguard_dolly_250 | 123 | 250 | 0.9801 | 0.9850 | 0.2933 | 0.1394 | 0.9852 | ff054230ce3c6a0c1de5cf9ed9f5a2c9db149df32a415cbcceed3f4d767639f5 | 68 | 0.0000 | 140 | 0.2071 |
| source | provguard_oasst1_250 | 13 | 250 | 0.9812 | 0.9836 | 0.2104 | 0.0481 | 0.9852 | 3531fb9ed26274a95ed06b3a7650b085d94a4cde8d2c1c15960a0de4269e665b | 68 | 0.0294 | 140 | 0.0571 |
| source | provguard_oasst1_250 | 42 | 250 | 0.9822 | 0.9860 | 0.2809 | 0.0529 | 0.9877 | f9c6f0adde1312bd5f0e79aedacf85015448bc11ca0c95eea05bde72add0b528 | 68 | 0.0147 | 140 | 0.0714 |
| source | provguard_oasst1_250 | 123 | 250 | 0.9824 | 0.9860 | 0.1869 | 0.0433 | 0.9877 | b320e78aefac3c29257f6da8a29a683de6dc7911d79333beb8410cc760c1fda4 | 68 | 0.0147 | 140 | 0.0571 |

| family | comparison | evaluation_source | training_seed_n | seeds | evaluation_n | absolute_fpr_reduction | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_repetitions | analysis |
|---|---|---|---|---|---|---|---|---|---|---|
| source | baseline vs provguard_dolly_250 | oasst1 | 3 | 13,42,123 | 140 | 0.0452 | -0.0167 | 0.1071 | 10000 | exploratory crossed training-seed/shared-prompt bootstrap; unadjusted pointwise CI |
| source | baseline vs provguard_oasst1_250 | dolly | 3 | 13,42,123 | 68 | 0.0539 | 0.0049 | 0.1176 | 10000 | exploratory crossed training-seed/shared-prompt bootstrap; unadjusted pointwise CI |

DeBERTa uses fixed max length 256, epochs 2, LR 5e-5, effective batch 16 (microbatch 4, accumulation 4), weight decay 0.01, max gradient norm 1.0, original-validation-F1 checkpoint selection and threshold 0.5 for both arms. These are secondary configuration choices, not method-specific tuning.

Source-isolated models use equal K=250 from the already approved original hard set, with DistilBERT primary training settings and seeds 13,42,123. Dolly training is evaluated on 140 OASST external HB prompts; OASST training on 68 Dolly prompts. Small source subsets and three seeds limit inference.

Additional independent random-control samples require new two-pass review from the full candidate pool; they are not synthesized from a hardness-biased reviewed subset.
