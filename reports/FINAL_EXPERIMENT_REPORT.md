# ProvGuard experiment report

Required runs completed: 30/30. Status: COMPLETE. Generated from verified prediction files at threshold 0.5. No paper was written.

## 1. Completion matrix

| Method | 13 | 42 | 123 | 2024 | 7777 |
|---|---|---|---|---|---|
| baseline | COMPLETED | COMPLETED | COMPLETED | COMPLETED | COMPLETED |
| curated | COMPLETED | COMPLETED | COMPLETED | COMPLETED | COMPLETED |
| random_real | COMPLETED | COMPLETED | COMPLETED | COMPLETED | COMPLETED |
| provguard_1000 | COMPLETED | COMPLETED | COMPLETED | COMPLETED | COMPLETED |
| provguard_250 | COMPLETED | COMPLETED | COMPLETED | COMPLETED | COMPLETED |
| provguard_500 | COMPLETED | COMPLETED | COMPLETED | COMPLETED | COMPLETED |

## 2. Main result table

| Method | Seeds | iid_f1 | hard_benign_fpr | external_hb_fpr | iid_recall | obfuscated_recall | domain_ood_f1 | structural_ood_fpr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5/5 | 0.9815 ± 0.0017 | 0.4158 ± 0.1170 | 0.2067 ± 0.0350 | 0.9871 ± 0.0047 | 0.9857 ± 0.0119 | 0.9033 ± 0.0409 | 0.7748 ± 0.1033 |
| curated | 5/5 | 0.9828 ± 0.0020 | 0.0587 ± 0.0163 | 0.1846 ± 0.0341 | 0.9892 ± 0.0017 | 0.9921 ± 0.0011 | 0.9131 ± 0.0348 | 0.5359 ± 0.1259 |
| random_real | 5/5 | 0.9840 ± 0.0009 | 0.2015 ± 0.0638 | 0.0731 ± 0.0104 | 0.9877 ± 0.0013 | 0.9891 ± 0.0028 | 0.8899 ± 0.0115 | 0.4356 ± 0.1309 |
| provguard_1000 | 5/5 | 0.9826 ± 0.0010 | 0.1384 ± 0.0465 | 0.0423 ± 0.0053 | 0.9869 ± 0.0016 | 0.9877 ± 0.0039 | 0.8839 ± 0.0090 | 0.3754 ± 0.1203 |

Means and sample SDs use every completed seed. Rows below five seeds are incomplete and cannot support final controlled claims. Hard-benign coverage: 808 usable public texts (208 external, 600 curated); 664 withheld rows excluded.

## 3. Raw per-seed table

| method | seed | augmentation_size | iid_f1 | hard_benign_fpr | external_hb_fpr | iid_recall | obfuscated_recall | domain_ood_f1 | structural_ood_fpr |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 13 | 0 | 0.9808 | 0.4641 | 0.2212 | 0.9894 | 0.9901 | 0.9143 | 0.7918 |
| baseline | 42 | 0 | 0.9787 | 0.2364 | 0.1442 | 0.9787 | 0.9654 | 0.8321 | 0.6106 |
| baseline | 123 | 0 | 0.9825 | 0.5223 | 0.2212 | 0.9894 | 0.9951 | 0.9227 | 0.8569 |
| baseline | 2024 | 0 | 0.9830 | 0.3626 | 0.2212 | 0.9899 | 0.9926 | 0.9356 | 0.7497 |
| baseline | 7777 | 0 | 0.9822 | 0.4938 | 0.2260 | 0.9884 | 0.9852 | 0.9120 | 0.8649 |
| curated | 13 | 343 | 0.9820 | 0.0334 | 0.1250 | 0.9894 | 0.9926 | 0.9233 | 0.5245 |
| curated | 42 | 343 | 0.9853 | 0.0557 | 0.2019 | 0.9908 | 0.9926 | 0.9445 | 0.4925 |
| curated | 123 | 343 | 0.9808 | 0.0582 | 0.2067 | 0.9889 | 0.9926 | 0.9198 | 0.4004 |
| curated | 2024 | 343 | 0.9844 | 0.0730 | 0.2019 | 0.9903 | 0.9926 | 0.9247 | 0.7427 |
| curated | 7777 | 343 | 0.9813 | 0.0730 | 0.1875 | 0.9865 | 0.9901 | 0.8534 | 0.5195 |
| random_real | 13 | 1000 | 0.9827 | 0.1671 | 0.0817 | 0.9860 | 0.9877 | 0.8726 | 0.2943 |
| random_real | 42 | 1000 | 0.9839 | 0.2847 | 0.0817 | 0.9870 | 0.9852 | 0.8853 | 0.6266 |
| random_real | 123 | 1000 | 0.9844 | 0.2191 | 0.0769 | 0.9879 | 0.9901 | 0.9002 | 0.4655 |
| random_real | 2024 | 1000 | 0.9839 | 0.1151 | 0.0577 | 0.9884 | 0.9926 | 0.9002 | 0.4595 |
| random_real | 7777 | 1000 | 0.9851 | 0.2215 | 0.0673 | 0.9894 | 0.9901 | 0.8910 | 0.3323 |
| provguard_1000 | 13 | 1000 | 0.9824 | 0.1337 | 0.0433 | 0.9841 | 0.9827 | 0.8873 | 0.2543 |
| provguard_1000 | 42 | 1000 | 0.9841 | 0.0866 | 0.0337 | 0.9870 | 0.9926 | 0.8705 | 0.3724 |
| provguard_1000 | 123 | 1000 | 0.9815 | 0.2116 | 0.0433 | 0.9879 | 0.9901 | 0.8953 | 0.4284 |
| provguard_1000 | 2024 | 1000 | 0.9820 | 0.1448 | 0.0481 | 0.9874 | 0.9877 | 0.8832 | 0.5485 |
| provguard_1000 | 7777 | 1000 | 0.9832 | 0.1151 | 0.0433 | 0.9879 | 0.9852 | 0.8832 | 0.2733 |
| provguard_250 | 13 | 250 | 0.9797 | 0.0668 | 0.0240 | 0.9778 | 0.9778 | 0.8619 | 0.2312 |
| provguard_250 | 42 | 250 | 0.9849 | 0.1374 | 0.0625 | 0.9899 | 0.9901 | 0.8902 | 0.3774 |
| provguard_250 | 123 | 250 | 0.9824 | 0.1485 | 0.0673 | 0.9865 | 0.9877 | 0.8744 | 0.4775 |
| provguard_250 | 2024 | 250 | 0.9844 | 0.0990 | 0.0481 | 0.9884 | 0.9901 | 0.8891 | 0.4885 |
| provguard_250 | 7777 | 250 | 0.9822 | 0.1683 | 0.0673 | 0.9845 | 0.9877 | 0.8860 | 0.4024 |
| provguard_500 | 13 | 500 | 0.9822 | 0.1795 | 0.0288 | 0.9860 | 0.9877 | 0.8945 | 0.4665 |
| provguard_500 | 42 | 500 | 0.9824 | 0.1238 | 0.0288 | 0.9831 | 0.9852 | 0.8730 | 0.3003 |
| provguard_500 | 123 | 500 | 0.9820 | 0.2364 | 0.0385 | 0.9874 | 0.9901 | 0.9059 | 0.3804 |
| provguard_500 | 2024 | 500 | 0.9832 | 0.0941 | 0.0288 | 0.9874 | 0.9926 | 0.8967 | 0.3183 |
| provguard_500 | 7777 | 500 | 0.9823 | 0.2104 | 0.0481 | 0.9903 | 0.9901 | 0.8963 | 0.3473 |

## 4. Ablation table

| Method | Seeds | iid_f1 | iid_recall | hard_benign_fpr | external_hb_fpr | obfuscated_recall |
|---|---:|---:|---:|---:|---:|---:|
| provguard_250 | 5/5 | 0.9827 ± 0.0021 | 0.9854 ± 0.0047 | 0.1240 ± 0.0407 | 0.0538 ± 0.0184 | 0.9867 ± 0.0051 |
| provguard_500 | 5/5 | 0.9824 ± 0.0005 | 0.9869 ± 0.0026 | 0.1688 ± 0.0592 | 0.0346 ± 0.0086 | 0.9891 ± 0.0028 |
| provguard_1000 | 5/5 | 0.9826 ± 0.0010 | 0.9869 ± 0.0016 | 0.1384 ± 0.0465 | 0.0423 ± 0.0053 | 0.9877 ± 0.0039 |

## 5. Hardness validation

| method | n | min | q1 | median | mean | q3 | max | n_above_0.5 | n_above_0.25 | n_above_0.1 | n_above_0.05 | n_above_0.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| provguard_1000 | 1000 | 0.004750 | 0.006894 | 0.016738 | 0.198910 | 0.330759 | 0.999864 | 172 | 294 | 359 | 401 | 612 |
| random_real_1000 | 1000 | 0.000340 | 0.000527 | 0.000903 | 0.058424 | 0.003713 | 0.999396 | 53 | 84 | 96 | 104 | 145 |
| curated | 343 | 0.000374 | 0.047005 | 0.651523 | 0.493198 | 0.670925 | 0.999922 | 195 | 246 | 251 | 256 | 266 |

Selection uses baseline seeds 13,42,123, descending mean score and two independent full-text semantic passes. Rank prefix checked through 1004. Random seed 42 draws fixed permutations independently of hardness within source/length strata. Overlap: 213. Nested sizes: 250,500,1000. Review is automated semantic adjudication, not human annotation.

## 6. Statistical results

| comparison | subset | seed | n | training_seed_n | FPR_A | FPR_B | delta_FPR | fp_fixed | new_fp | discordant_n | mcnemar_statistic | p_value | p_holm | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_repetitions | bootstrap_seed | analysis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline vs provguard_1000 | usable_hard_benign | 13 | 808 | 1 | 0.464109 | 0.133663 | 0.330446 | 295.000000 | 28.000000 | 323.000000 | 220.708978 | 0.000000 | 0.000000 | 0.293317 | 0.368812 | 10000 | 424213 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | usable_hard_benign | 42 | 808 | 1 | 0.236386 | 0.086634 | 0.149752 | 130.000000 | 9.000000 | 139.000000 | 105.330935 | 0.000000 | 0.000000 | 0.123762 | 0.176980 | 10000 | 424242 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | usable_hard_benign | 123 | 808 | 1 | 0.522277 | 0.211634 | 0.310644 | 283.000000 | 32.000000 | 315.000000 | 200.003175 | 0.000000 | 0.000000 | 0.273515 | 0.347772 | 10000 | 424323 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | usable_hard_benign | 2024 | 808 | 1 | 0.362624 | 0.144802 | 0.217822 | 211.000000 | 35.000000 | 246.000000 | 125.918699 | 0.000000 | 0.000000 | 0.184406 | 0.253713 | 10000 | 426224 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | usable_hard_benign | 7777 | 808 | 1 | 0.493812 | 0.115099 | 0.378713 | 306.000000 | 0.000000 | 306.000000 | 306.000000 | 0.000000 | 0.000000 | 0.345297 | 0.412129 | 10000 | 431977 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | usable_hard_benign | AGGREGATE | 808 | 5 | 0.415842 | 0.138366 | 0.277475 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | 0.200743 | 0.351238 | 10000 | 424242 | crossed bootstrap resamples paired training seeds and shared prompt IDs; no pooled McNemar p-value; all five seeds |
| baseline vs provguard_1000 | external_hard_benign | 13 | 208 | 1 | 0.221154 | 0.043269 | 0.177885 | 37.000000 | 0.000000 | 37.000000 | 37.000000 | 0.000000 | 0.000000 | 0.129808 | 0.230769 | 10000 | 424213 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | external_hard_benign | 42 | 208 | 1 | 0.144231 | 0.033654 | 0.110577 | 23.000000 | 0.000000 | 23.000000 | 23.000000 | 0.000000 | 0.000003 | 0.072115 | 0.153846 | 10000 | 424242 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | external_hard_benign | 123 | 208 | 1 | 0.221154 | 0.043269 | 0.177885 | 37.000000 | 0.000000 | 37.000000 | 37.000000 | 0.000000 | 0.000000 | 0.125000 | 0.230769 | 10000 | 424323 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | external_hard_benign | 2024 | 208 | 1 | 0.221154 | 0.048077 | 0.173077 | 36.000000 | 0.000000 | 36.000000 | 36.000000 | 0.000000 | 0.000000 | 0.125000 | 0.225962 | 10000 | 426224 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | external_hard_benign | 7777 | 208 | 1 | 0.225962 | 0.043269 | 0.182692 | 38.000000 | 0.000000 | 38.000000 | 38.000000 | 0.000000 | 0.000000 | 0.134615 | 0.235577 | 10000 | 431977 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| baseline vs provguard_1000 | external_hard_benign | AGGREGATE | 208 | 5 | 0.206731 | 0.042308 | 0.164423 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | 0.117308 | 0.214423 | 10000 | 424242 | crossed bootstrap resamples paired training seeds and shared prompt IDs; no pooled McNemar p-value; all five seeds |
| random_real vs provguard_1000 | usable_hard_benign | 13 | 808 | 1 | 0.167079 | 0.133663 | 0.033416 | 56.000000 | 29.000000 | 85.000000 | 8.576471 | 0.004512 | 0.036093 | 0.011139 | 0.055693 | 10000 | 424213 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | usable_hard_benign | 42 | 808 | 1 | 0.284653 | 0.086634 | 0.198020 | 160.000000 | 0.000000 | 160.000000 | 160.000000 | 0.000000 | 0.000000 | 0.170792 | 0.226485 | 10000 | 424242 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | usable_hard_benign | 123 | 808 | 1 | 0.219059 | 0.211634 | 0.007426 | 65.000000 | 59.000000 | 124.000000 | 0.290323 | 0.653598 | 1.000000 | -0.019802 | 0.034653 | 10000 | 424323 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | usable_hard_benign | 2024 | 808 | 1 | 0.115099 | 0.144802 | -0.029703 | 23.000000 | 47.000000 | 70.000000 | 8.228571 | 0.005583 | 0.039079 | -0.050743 | -0.009901 | 10000 | 426224 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | usable_hard_benign | 7777 | 808 | 1 | 0.221535 | 0.115099 | 0.106436 | 88.000000 | 2.000000 | 90.000000 | 82.177778 | 0.000000 | 0.000000 | 0.085396 | 0.128713 | 10000 | 431977 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | usable_hard_benign | AGGREGATE | 808 | 5 | 0.201485 | 0.138366 | 0.063119 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | -0.004950 | 0.140842 | 10000 | 424242 | crossed bootstrap resamples paired training seeds and shared prompt IDs; no pooled McNemar p-value; all five seeds |
| random_real vs provguard_1000 | external_hard_benign | 13 | 208 | 1 | 0.081731 | 0.043269 | 0.038462 | 10.000000 | 2.000000 | 12.000000 | 5.333333 | 0.038574 | 0.192871 | 0.009615 | 0.072115 | 10000 | 424213 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | external_hard_benign | 42 | 208 | 1 | 0.081731 | 0.033654 | 0.048077 | 10.000000 | 0.000000 | 10.000000 | 10.000000 | 0.001953 | 0.018667 | 0.019231 | 0.076923 | 10000 | 424242 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | external_hard_benign | 123 | 208 | 1 | 0.076923 | 0.043269 | 0.033654 | 9.000000 | 2.000000 | 11.000000 | 4.454545 | 0.065430 | 0.261719 | 0.004808 | 0.067308 | 10000 | 424323 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | external_hard_benign | 2024 | 208 | 1 | 0.057692 | 0.048077 | 0.009615 | 3.000000 | 1.000000 | 4.000000 | 1.000000 | 0.625000 | 1.000000 | -0.009615 | 0.028846 | 10000 | 426224 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | external_hard_benign | 7777 | 208 | 1 | 0.067308 | 0.043269 | 0.024038 | 6.000000 | 1.000000 | 7.000000 | 3.571429 | 0.125000 | 0.375000 | 0.000000 | 0.048077 | 10000 | 431977 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| random_real vs provguard_1000 | external_hard_benign | AGGREGATE | 208 | 5 | 0.073077 | 0.042308 | 0.030769 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | 0.009615 | 0.054808 | 10000 | 424242 | crossed bootstrap resamples paired training seeds and shared prompt IDs; no pooled McNemar p-value; all five seeds |
| curated vs provguard_1000 | usable_hard_benign | 13 | 808 | 1 | 0.033416 | 0.133663 | -0.100248 | 19.000000 | 100.000000 | 119.000000 | 55.134454 | 0.000000 | 0.000000 | -0.126238 | -0.074257 | 10000 | 424213 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | usable_hard_benign | 42 | 808 | 1 | 0.055693 | 0.086634 | -0.030941 | 35.000000 | 60.000000 | 95.000000 | 6.578947 | 0.013379 | 0.080272 | -0.054455 | -0.007426 | 10000 | 424242 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | usable_hard_benign | 123 | 808 | 1 | 0.058168 | 0.211634 | -0.153465 | 36.000000 | 160.000000 | 196.000000 | 78.448980 | 0.000000 | 0.000000 | -0.185644 | -0.120050 | 10000 | 424323 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | usable_hard_benign | 2024 | 808 | 1 | 0.073020 | 0.144802 | -0.071782 | 39.000000 | 97.000000 | 136.000000 | 24.735294 | 0.000001 | 0.000008 | -0.100248 | -0.044554 | 10000 | 426224 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | usable_hard_benign | 7777 | 808 | 1 | 0.073020 | 0.115099 | -0.042079 | 40.000000 | 74.000000 | 114.000000 | 10.140351 | 0.001867 | 0.018667 | -0.068069 | -0.017327 | 10000 | 431977 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | usable_hard_benign | AGGREGATE | 808 | 5 | 0.058663 | 0.138366 | -0.079703 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | -0.126733 | -0.037871 | 10000 | 424242 | crossed bootstrap resamples paired training seeds and shared prompt IDs; no pooled McNemar p-value; all five seeds |
| curated vs provguard_1000 | external_hard_benign | 13 | 208 | 1 | 0.125000 | 0.043269 | 0.081731 | 19.000000 | 2.000000 | 21.000000 | 13.761905 | 0.000221 | 0.002434 | 0.043269 | 0.125000 | 10000 | 424213 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | external_hard_benign | 42 | 208 | 1 | 0.201923 | 0.033654 | 0.168269 | 35.000000 | 0.000000 | 35.000000 | 35.000000 | 0.000000 | 0.000000 | 0.120192 | 0.221154 | 10000 | 424242 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | external_hard_benign | 123 | 208 | 1 | 0.206731 | 0.043269 | 0.163462 | 36.000000 | 2.000000 | 38.000000 | 30.421053 | 0.000000 | 0.000000 | 0.110577 | 0.221154 | 10000 | 424323 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | external_hard_benign | 2024 | 208 | 1 | 0.201923 | 0.048077 | 0.153846 | 32.000000 | 0.000000 | 32.000000 | 32.000000 | 0.000000 | 0.000000 | 0.105769 | 0.206731 | 10000 | 426224 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | external_hard_benign | 7777 | 208 | 1 | 0.187500 | 0.043269 | 0.144231 | 31.000000 | 1.000000 | 32.000000 | 28.125000 | 0.000000 | 0.000000 | 0.096154 | 0.197115 | 10000 | 431977 | paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard |
| curated vs provguard_1000 | external_hard_benign | AGGREGATE | 208 | 5 | 0.184615 | 0.042308 | 0.142308 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | 0.093269 | 0.193269 | 10000 | 424242 | crossed bootstrap resamples paired training seeds and shared prompt IDs; no pooled McNemar p-value; all five seeds |

Positive delta means comparator FPR minus ProvGuard FPR. Exact McNemar tests are per seed with Holm correction. Aggregate 10,000-resample crossed bootstrap resamples training seeds and shared prompts; predictions across seeds are never pooled as independent examples. CIs are pointwise 95%.

## 7. Error analysis

ERROR_TRANSITIONS.csv includes all aligned rows, including remaining errors and unchanged correct rows. ERROR_CATEGORIES.csv maps released curated source taxonomy to categories and uses explicit ordered lexical proxies for real prompts. Categorization is exploratory, automated, and separate from semantic validity adjudication.

## 8. H1/H2/H3 verdicts

H1: SUPPORTED. H2 (primary novelty test): INCONCLUSIVE. H3: SUPPORTED.
Final outcome: INCONCLUSIVE. Practical project criterion satisfied: True.

H1/H2 support conservatively requires positive aggregate lower CI bounds for both usable and external HB; mixed endpoint evidence is inconclusive, with each endpoint verdict saved separately. H3 compares paired seed-bootstrap IID F1/recall changes against predefined loss bounds 0.01/0.02. A confirmed obfuscated-recall decline vetoes preservation; a negative point estimate with uncertain sign is inconclusive. No obfuscated noninferiority margin was predefined, so no formal margin claim is made. Five training seeds give limited uncertainty resolution.

## 9. Security-usability interpretation

| comparison | matched_seed_n | matched_seeds | status | hard_benign_fpr_absolute_reduction | hard_benign_fpr_relative_reduction | external_hb_fpr_absolute_reduction | external_hb_fpr_relative_reduction | iid_f1_provguard_minus_comparator | iid_recall_provguard_minus_comparator | obfuscated_recall_provguard_minus_comparator | domain_ood_f1_provguard_minus_comparator | structural_ood_fpr_provguard_minus_comparator |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline vs provguard_1000 | 5 | 13,42,123,2024,7777 | COMPLETE | 0.277475 | 0.667262 | 0.164423 | 0.795349 | 0.001179 | -0.000290 | 0.001975 | -0.019448 | -0.399399 |
| random_real vs provguard_1000 | 5 | 13,42,123,2024,7777 | COMPLETE | 0.063119 | 0.313268 | 0.030769 | 0.421053 | -0.001338 | -0.000870 | -0.001481 | -0.005978 | -0.060260 |
| curated vs provguard_1000 | 5 | 13,42,123,2024,7777 | COMPLETE | -0.079703 | -1.358650 | 0.142308 | 0.770833 | -0.000135 | -0.002319 | -0.004444 | -0.029244 | -0.160561 |

ProvGuard versus equal-size matched Random-Real is the primary selection comparison. Curated-HN is secondary and has a different size. Public stress-set FPR is not a deployment prevalence estimate.

## 10. Best configuration

Primary configuration remains preregistered K=1000, DistilBERT, max length 256, epochs 2, effective batch 16, LR 5e-5, weight decay 0.01, max gradient norm 1.0. Original validation F1 selects the checkpoint. No preferred K is chosen from test FPR alone.

## 11. conference readiness

Core experiment matrix and outputs complete; review the measured hypothesis verdicts and fallible automated adjudication before paper writing.

Architecture transfer, source transfer and additional random samples are secondary. No such result is claimed without a separate measured output. External HB source labels identify 68 Dolly and 140 OASST texts. SOURCE_STRATIFIED_RESULTS.csv reports mixed-source training performance separately on these populations; source-isolated training is a distinct secondary confirmation.

## 12. Output paths and reproducibility

`results_master.csv`; `reports/{PRIMARY_RESULTS,ABLATION_RESULTS,STATISTICAL_RESULTS,HARDNESS_ANALYSIS,ERROR_TRANSITIONS,ERROR_CATEGORIES,COMPARISON_RESULTS}.csv`; `reports/experiment_registry.json`; `reports/environment_report.json`; `reports/canonical_leakage_audit.json`; `reports/semantic_adjudication_audit.json`; `reports/candidate_selection_audit.json`; `figures/final/`.

Exact commands and logs are in the registry. Source hashes and packages are in environment_report.json. Per-arm configs bind augmentation files and original frozen train/validation hashes. Canonical leakage TF-IDF char 3–5 representation is persisted; threshold 0.90 is unchanged. Curated count is the valid filtered count without duplicates. Upstream commit: 7984e2d6d95e98e914d2b0cd92574b1752614b10.
