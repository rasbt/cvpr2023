# 00_pytorch-vit-random-init.py (finetuning not pretrained)

Train a plain Vision transformer from scratch.


```
Python version       : 3.9.16
IPython version      : 8.12.0

torch    : 2.1.0.dev20230612+cu118
lightning: 2.1.0.dev0

Epoch: 0001/0010 | Batch 0000/2812 | Loss: 2.5971
Epoch: 0001/0010 | Batch 0300/2812 | Loss: 1.9497
Epoch: 0001/0010 | Batch 0600/2812 | Loss: 1.6241
Epoch: 0001/0010 | Batch 0900/2812 | Loss: 2.0443
Epoch: 0001/0010 | Batch 1200/2812 | Loss: 1.9395
Epoch: 0001/0010 | Batch 1500/2812 | Loss: 1.6568
Epoch: 0001/0010 | Batch 1800/2812 | Loss: 1.7207
Epoch: 0001/0010 | Batch 2100/2812 | Loss: 1.6199
Epoch: 0001/0010 | Batch 2400/2812 | Loss: 1.1766
Epoch: 0001/0010 | Batch 2700/2812 | Loss: 1.5693
Epoch: 0001/0010 | Train acc.: 34.81% | Val acc.: 47.22%
Epoch: 0002/0010 | Batch 0000/2812 | Loss: 1.0045
Epoch: 0002/0010 | Batch 0300/2812 | Loss: 1.5517
Epoch: 0002/0010 | Batch 0600/2812 | Loss: 1.5892
Epoch: 0002/0010 | Batch 0900/2812 | Loss: 1.3893
Epoch: 0002/0010 | Batch 1200/2812 | Loss: 1.2300
Epoch: 0002/0010 | Batch 1500/2812 | Loss: 0.9359
Epoch: 0002/0010 | Batch 1800/2812 | Loss: 1.4835
Epoch: 0002/0010 | Batch 2100/2812 | Loss: 1.7854
Epoch: 0002/0010 | Batch 2400/2812 | Loss: 0.9467
Epoch: 0002/0010 | Batch 2700/2812 | Loss: 1.7041
Epoch: 0002/0010 | Train acc.: 50.24% | Val acc.: 52.48%
Epoch: 0003/0010 | Batch 0000/2812 | Loss: 1.3514
Epoch: 0003/0010 | Batch 0300/2812 | Loss: 1.2390
Epoch: 0003/0010 | Batch 0600/2812 | Loss: 1.0620
Epoch: 0003/0010 | Batch 0900/2812 | Loss: 1.3221
Epoch: 0003/0010 | Batch 1200/2812 | Loss: 0.8957
Epoch: 0003/0010 | Batch 1500/2812 | Loss: 0.7106
Epoch: 0003/0010 | Batch 1800/2812 | Loss: 1.0991
Epoch: 0003/0010 | Batch 2100/2812 | Loss: 1.1103
Epoch: 0003/0010 | Batch 2400/2812 | Loss: 1.1114
Epoch: 0003/0010 | Batch 2700/2812 | Loss: 1.2009
Epoch: 0003/0010 | Train acc.: 55.98% | Val acc.: 55.98%
Epoch: 0004/0010 | Batch 0000/2812 | Loss: 1.3007
Epoch: 0004/0010 | Batch 0300/2812 | Loss: 1.0196
Epoch: 0004/0010 | Batch 0600/2812 | Loss: 0.9170
Epoch: 0004/0010 | Batch 0900/2812 | Loss: 1.1119
Epoch: 0004/0010 | Batch 1200/2812 | Loss: 1.4091
Epoch: 0004/0010 | Batch 1500/2812 | Loss: 0.9497
Epoch: 0004/0010 | Batch 1800/2812 | Loss: 1.1372
Epoch: 0004/0010 | Batch 2100/2812 | Loss: 1.5214
Epoch: 0004/0010 | Batch 2400/2812 | Loss: 1.1381
Epoch: 0004/0010 | Batch 2700/2812 | Loss: 1.1540
Epoch: 0004/0010 | Train acc.: 59.33% | Val acc.: 59.14%
Epoch: 0005/0010 | Batch 0000/2812 | Loss: 1.3431
Epoch: 0005/0010 | Batch 0300/2812 | Loss: 1.5235
Epoch: 0005/0010 | Batch 0600/2812 | Loss: 0.6558
Epoch: 0005/0010 | Batch 0900/2812 | Loss: 1.3207
Epoch: 0005/0010 | Batch 1200/2812 | Loss: 1.1548
Epoch: 0005/0010 | Batch 1500/2812 | Loss: 0.7654
Epoch: 0005/0010 | Batch 1800/2812 | Loss: 0.7152
Epoch: 0005/0010 | Batch 2100/2812 | Loss: 1.2607
Epoch: 0005/0010 | Batch 2400/2812 | Loss: 0.9752
Epoch: 0005/0010 | Batch 2700/2812 | Loss: 1.2946
Epoch: 0005/0010 | Train acc.: 62.04% | Val acc.: 60.06%
Epoch: 0006/0010 | Batch 0000/2812 | Loss: 0.9951
Epoch: 0006/0010 | Batch 0300/2812 | Loss: 1.1221
Epoch: 0006/0010 | Batch 0600/2812 | Loss: 0.4458
Epoch: 0006/0010 | Batch 0900/2812 | Loss: 1.2516
Epoch: 0006/0010 | Batch 1200/2812 | Loss: 0.6722
Epoch: 0006/0010 | Batch 1500/2812 | Loss: 1.0663
Epoch: 0006/0010 | Batch 1800/2812 | Loss: 0.9296
Epoch: 0006/0010 | Batch 2100/2812 | Loss: 0.7549
Epoch: 0006/0010 | Batch 2400/2812 | Loss: 0.6679
Epoch: 0006/0010 | Batch 2700/2812 | Loss: 1.2472
Epoch: 0006/0010 | Train acc.: 64.22% | Val acc.: 61.80%
Epoch: 0007/0010 | Batch 0000/2812 | Loss: 1.2649
Epoch: 0007/0010 | Batch 0300/2812 | Loss: 0.5699
Epoch: 0007/0010 | Batch 0600/2812 | Loss: 1.1111
Epoch: 0007/0010 | Batch 0900/2812 | Loss: 1.4958
Epoch: 0007/0010 | Batch 1200/2812 | Loss: 0.9095
Epoch: 0007/0010 | Batch 1500/2812 | Loss: 0.6518
Epoch: 0007/0010 | Batch 1800/2812 | Loss: 1.2235
Epoch: 0007/0010 | Batch 2100/2812 | Loss: 0.9390
Epoch: 0007/0010 | Batch 2400/2812 | Loss: 1.3117
Epoch: 0007/0010 | Batch 2700/2812 | Loss: 0.7095
Epoch: 0007/0010 | Train acc.: 66.12% | Val acc.: 61.14%
Epoch: 0008/0010 | Batch 0000/2812 | Loss: 0.5717
Epoch: 0008/0010 | Batch 0300/2812 | Loss: 0.9590
Epoch: 0008/0010 | Batch 0600/2812 | Loss: 1.2407
Epoch: 0008/0010 | Batch 0900/2812 | Loss: 0.6916
Epoch: 0008/0010 | Batch 1200/2812 | Loss: 0.6023
Epoch: 0008/0010 | Batch 1500/2812 | Loss: 0.6515
Epoch: 0008/0010 | Batch 1800/2812 | Loss: 0.6961
Epoch: 0008/0010 | Batch 2100/2812 | Loss: 0.5924
Epoch: 0008/0010 | Batch 2400/2812 | Loss: 0.7415
Epoch: 0008/0010 | Batch 2700/2812 | Loss: 0.9263
Epoch: 0008/0010 | Train acc.: 68.04% | Val acc.: 63.60%
Epoch: 0009/0010 | Batch 0000/2812 | Loss: 0.9797
Epoch: 0009/0010 | Batch 0300/2812 | Loss: 0.5473
Epoch: 0009/0010 | Batch 0600/2812 | Loss: 0.7215
Epoch: 0009/0010 | Batch 0900/2812 | Loss: 1.2585
Epoch: 0009/0010 | Batch 1200/2812 | Loss: 0.6315
Epoch: 0009/0010 | Batch 1500/2812 | Loss: 0.5185
Epoch: 0009/0010 | Batch 1800/2812 | Loss: 0.7371
Epoch: 0009/0010 | Batch 2100/2812 | Loss: 1.3996
Epoch: 0009/0010 | Batch 2400/2812 | Loss: 0.5328
Epoch: 0009/0010 | Batch 2700/2812 | Loss: 1.0833
Epoch: 0009/0010 | Train acc.: 69.50% | Val acc.: 65.32%
Epoch: 0010/0010 | Batch 0000/2812 | Loss: 0.6748
Epoch: 0010/0010 | Batch 0300/2812 | Loss: 0.7661
Epoch: 0010/0010 | Batch 0600/2812 | Loss: 0.8363
Epoch: 0010/0010 | Batch 0900/2812 | Loss: 0.4878
Epoch: 0010/0010 | Batch 1200/2812 | Loss: 1.4964
Epoch: 0010/0010 | Batch 1500/2812 | Loss: 1.4705
Epoch: 0010/0010 | Batch 1800/2812 | Loss: 1.1050
Epoch: 0010/0010 | Batch 2100/2812 | Loss: 0.4973
Epoch: 0010/0010 | Batch 2400/2812 | Loss: 0.5641
Epoch: 0010/0010 | Batch 2700/2812 | Loss: 0.6933
Epoch: 0010/0010 | Train acc.: 71.16% | Val acc.: 62.80%
Time elapsed 61.48 min
Memory used: 3.71 GB
Test accuracy 62.85%
```

# 01_pytorch-vit.py (finetuning pretrained)

Like above but using a pretrained vision transformer.

```
Epoch: 0001/0003 | Batch 0000/2812 | Loss: 2.4934
Epoch: 0001/0003 | Batch 0300/2812 | Loss: 0.0954
Epoch: 0001/0003 | Batch 0600/2812 | Loss: 0.0981
Epoch: 0001/0003 | Batch 0900/2812 | Loss: 0.2078
Epoch: 0001/0003 | Batch 1200/2812 | Loss: 0.3588
Epoch: 0001/0003 | Batch 1500/2812 | Loss: 0.0104
Epoch: 0001/0003 | Batch 1800/2812 | Loss: 0.1560
Epoch: 0001/0003 | Batch 2100/2812 | Loss: 0.0474
Epoch: 0001/0003 | Batch 2400/2812 | Loss: 0.4250
Epoch: 0001/0003 | Batch 2700/2812 | Loss: 0.4414
Epoch: 0001/0003 | Train acc.: 92.40% | Val acc.: 94.12%
Epoch: 0002/0003 | Batch 0000/2812 | Loss: 0.0912
Epoch: 0002/0003 | Batch 0300/2812 | Loss: 0.0337
Epoch: 0002/0003 | Batch 0600/2812 | Loss: 0.1545
Epoch: 0002/0003 | Batch 0900/2812 | Loss: 0.0478
Epoch: 0002/0003 | Batch 1200/2812 | Loss: 0.0697
Epoch: 0002/0003 | Batch 1500/2812 | Loss: 0.1314
Epoch: 0002/0003 | Batch 1800/2812 | Loss: 0.2215
Epoch: 0002/0003 | Batch 2100/2812 | Loss: 0.4472
Epoch: 0002/0003 | Batch 2400/2812 | Loss: 0.0322
Epoch: 0002/0003 | Batch 2700/2812 | Loss: 0.1310
Epoch: 0002/0003 | Train acc.: 96.28% | Val acc.: 94.50%
Epoch: 0003/0003 | Batch 0000/2812 | Loss: 0.0902
Epoch: 0003/0003 | Batch 0300/2812 | Loss: 0.1597
Epoch: 0003/0003 | Batch 0600/2812 | Loss: 0.0106
Epoch: 0003/0003 | Batch 0900/2812 | Loss: 0.0032
Epoch: 0003/0003 | Batch 1200/2812 | Loss: 0.0147
Epoch: 0003/0003 | Batch 1500/2812 | Loss: 0.0082
Epoch: 0003/0003 | Batch 1800/2812 | Loss: 0.0078
Epoch: 0003/0003 | Batch 2100/2812 | Loss: 0.0060
Epoch: 0003/0003 | Batch 2400/2812 | Loss: 0.1395
Epoch: 0003/0003 | Batch 2700/2812 | Loss: 0.1128
Epoch: 0003/0003 | Train acc.: 97.24% | Val acc.: 95.74%
Time elapsed 18.70 min
Memory used: 3.71 GB
Test accuracy 95.37%
```

# 02_pytorch-vit-compile.py

Like above but with `torch.compile`.

```
Epoch: 0001/0003 | Batch 0000/2812 | Loss: 2.4934
Epoch: 0001/0003 | Batch 0300/2812 | Loss: 0.4464
Epoch: 0001/0003 | Batch 0600/2812 | Loss: 0.1263
Epoch: 0001/0003 | Batch 0900/2812 | Loss: 0.1233
Epoch: 0001/0003 | Batch 1200/2812 | Loss: 0.4541
Epoch: 0001/0003 | Batch 1500/2812 | Loss: 0.0186
Epoch: 0001/0003 | Batch 1800/2812 | Loss: 0.0930
Epoch: 0001/0003 | Batch 2100/2812 | Loss: 0.0396
Epoch: 0001/0003 | Batch 2400/2812 | Loss: 0.2211
Epoch: 0001/0003 | Batch 2700/2812 | Loss: 0.1570
Epoch: 0002/0003 | Batch 0000/2812 | Loss: 0.0186
Epoch: 0002/0003 | Batch 0300/2812 | Loss: 0.0337
Epoch: 0002/0003 | Batch 0600/2812 | Loss: 0.1992
Epoch: 0002/0003 | Batch 0900/2812 | Loss: 0.0275
Epoch: 0002/0003 | Batch 1200/2812 | Loss: 0.0874
Epoch: 0002/0003 | Batch 1500/2812 | Loss: 0.0739
Epoch: 0002/0003 | Batch 1800/2812 | Loss: 0.0432
Epoch: 0002/0003 | Batch 2100/2812 | Loss: 0.0564
Epoch: 0002/0003 | Batch 2400/2812 | Loss: 0.0110
Epoch: 0002/0003 | Batch 2700/2812 | Loss: 0.0948
Epoch: 0002/0003 | Train acc.: 96.15% | Val acc.: 94.72%
Epoch: 0003/0003 | Batch 0000/2812 | Loss: 0.0462
Epoch: 0003/0003 | Batch 0300/2812 | Loss: 0.1742
Epoch: 0003/0003 | Batch 0600/2812 | Loss: 0.0039
Epoch: 0003/0003 | Batch 0900/2812 | Loss: 0.0113
Epoch: 0003/0003 | Batch 1200/2812 | Loss: 0.0022
Epoch: 0003/0003 | Batch 1500/2812 | Loss: 0.0047
Epoch: 0003/0003 | Batch 1800/2812 | Loss: 0.0667
Epoch: 0003/0003 | Batch 2100/2812 | Loss: 0.0145
Epoch: 0003/0003 | Batch 2400/2812 | Loss: 0.0071
Epoch: 0003/0003 | Batch 2700/2812 | Loss: 0.0085
Epoch: 0003/0003 | Train acc.: 97.29% | Val acc.: 93.16%
Time elapsed 18.01 min
Memory used: 3.73 GB
Test accuracy 93.02%
```

# 03_fabric-vit.py

Like `01_pytorch-vit.py` but using Fabric.

```
Epoch: 0001/0003 | Batch 0000/2812 | Loss: 2.4934
Epoch: 0001/0003 | Batch 0300/2812 | Loss: 0.4444
Epoch: 0001/0003 | Batch 0600/2812 | Loss: 0.2839
Epoch: 0001/0003 | Batch 0900/2812 | Loss: 0.0866
Epoch: 0001/0003 | Batch 1200/2812 | Loss: 0.5020
Epoch: 0001/0003 | Batch 1500/2812 | Loss: 0.0164
Epoch: 0001/0003 | Batch 1800/2812 | Loss: 0.0770
Epoch: 0001/0003 | Batch 2100/2812 | Loss: 0.0157
Epoch: 0001/0003 | Batch 2400/2812 | Loss: 0.3030
Epoch: 0001/0003 | Batch 2700/2812 | Loss: 0.0797
Epoch: 0001/0003 | Train acc.: 92.25% | Val acc.: 93.22%
Epoch: 0002/0003 | Batch 0000/2812 | Loss: 0.0194
Epoch: 0002/0003 | Batch 0300/2812 | Loss: 0.0443
Epoch: 0002/0003 | Batch 0600/2812 | Loss: 0.1183
Epoch: 0002/0003 | Batch 0900/2812 | Loss: 0.0122
Epoch: 0002/0003 | Batch 1200/2812 | Loss: 0.0111
Epoch: 0002/0003 | Batch 1500/2812 | Loss: 0.0069
Epoch: 0002/0003 | Batch 1800/2812 | Loss: 0.0209
Epoch: 0002/0003 | Batch 2100/2812 | Loss: 0.5524
Epoch: 0002/0003 | Batch 2400/2812 | Loss: 0.2935
Epoch: 0002/0003 | Batch 2700/2812 | Loss: 0.1671
Epoch: 0002/0003 | Train acc.: 96.12% | Val acc.: 94.72%
Epoch: 0003/0003 | Batch 0000/2812 | Loss: 0.0296
Epoch: 0003/0003 | Batch 0300/2812 | Loss: 0.0707
Epoch: 0003/0003 | Batch 0600/2812 | Loss: 0.0027
Epoch: 0003/0003 | Batch 0900/2812 | Loss: 0.0081
Epoch: 0003/0003 | Batch 1200/2812 | Loss: 0.0087
Epoch: 0003/0003 | Batch 1500/2812 | Loss: 0.0014
Epoch: 0003/0003 | Batch 1800/2812 | Loss: 0.2059
Epoch: 0003/0003 | Batch 2100/2812 | Loss: 0.0106
Epoch: 0003/0003 | Batch 2400/2812 | Loss: 0.0675
^[[BEpoch: 0003/0003 | Batch 2700/2812 | Loss: 0.0527
Epoch: 0003/0003 | Train acc.: 97.35% | Val acc.: 95.48%
Time elapsed 18.49 min
Memory used: 3.71 GB
Test accuracy 95.78%
```

# 04_fabric-vit-mixed-precision.py

Like `03_fabric-vit.py` but with bfloat16 mixed precision training (`bf16-mixed`) -- if your GPU doesn't suppoirt it, you can change it to float16 mixed precision (`16-mixed`).

```
Epoch: 0001/0003 | Batch 0000/2812 | Loss: 2.4933
Epoch: 0001/0003 | Batch 0300/2812 | Loss: 0.1566
Epoch: 0001/0003 | Batch 0600/2812 | Loss: 0.1762
Epoch: 0001/0003 | Batch 0900/2812 | Loss: 0.1912
Epoch: 0001/0003 | Batch 1200/2812 | Loss: 0.2519
Epoch: 0001/0003 | Batch 1500/2812 | Loss: 0.0415
Epoch: 0001/0003 | Batch 1800/2812 | Loss: 0.0783
Epoch: 0001/0003 | Batch 2100/2812 | Loss: 0.0447
Epoch: 0001/0003 | Batch 2400/2812 | Loss: 0.5027
Epoch: 0001/0003 | Batch 2700/2812 | Loss: 0.3819
Epoch: 0001/0003 | Train acc.: 92.36% | Val acc.: 93.00%
Epoch: 0002/0003 | Batch 0000/2812 | Loss: 0.1281
Epoch: 0002/0003 | Batch 0300/2812 | Loss: 0.2139
Epoch: 0002/0003 | Batch 0600/2812 | Loss: 0.1355
Epoch: 0002/0003 | Batch 0900/2812 | Loss: 0.1130
Epoch: 0002/0003 | Batch 1200/2812 | Loss: 0.1395
Epoch: 0002/0003 | Batch 1500/2812 | Loss: 0.0121
Epoch: 0002/0003 | Batch 1800/2812 | Loss: 0.0389
Epoch: 0002/0003 | Batch 2100/2812 | Loss: 0.2634
Epoch: 0002/0003 | Batch 2400/2812 | Loss: 0.0625
Epoch: 0002/0003 | Batch 2700/2812 | Loss: 0.1037
Epoch: 0002/0003 | Train acc.: 96.20% | Val acc.: 95.26%
Epoch: 0003/0003 | Batch 0000/2812 | Loss: 0.0712
Epoch: 0003/0003 | Batch 0300/2812 | Loss: 0.1453
Epoch: 0003/0003 | Batch 0600/2812 | Loss: 0.0075
Epoch: 0003/0003 | Batch 0900/2812 | Loss: 0.0663
Epoch: 0003/0003 | Batch 1200/2812 | Loss: 0.0016
Epoch: 0003/0003 | Batch 1500/2812 | Loss: 0.0029
Epoch: 0003/0003 | Batch 1800/2812 | Loss: 0.0449
Epoch: 0003/0003 | Batch 2100/2812 | Loss: 0.0020
Epoch: 0003/0003 | Batch 2400/2812 | Loss: 0.2057
Epoch: 0003/0003 | Batch 2700/2812 | Loss: 0.0410
Epoch: 0003/0003 | Train acc.: 97.20% | Val acc.: 95.56%
Time elapsed 6.36 min
Memory used: 3.04 GB
Test accuracy 95.24%
```

# 05_fabric-vit-mixed-ddp.py

Like `04_fabric-vit-mixed-precision.py` but using Distributed Data Parallelism (DDP).

```
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.4056
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.2974
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.3568
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.4210
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.2212
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.1103
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.0332
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.1673
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.1595
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.2520
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.1233
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.0192
Epoch: 0001/0003 | Train acc.: 93.58% | Val acc.: 95.96%
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0078
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0372
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0221
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0254
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.1037
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0042
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0218
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0519
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.0038
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.0028
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.0067
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.0076
Epoch: 0002/0003 | Train acc.: 97.86% | Val acc.: 95.76%
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0073
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0024
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0173
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0155
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0133
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0012
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0024
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0185
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.1074
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.0068
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.0034
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.2559
Epoch: 0003/0003 | Train acc.: 98.58% | Val acc.: 95.82%
Time elapsed 2.19 min
Memory used: 4.09 GB
Test accuracy 95.73%
```

# 06_fabric-vit-mixed-fsdp.py

Like `05_fabric-vit-mixed-ddp.py` but using Fully Sharded Data Parallelism (FSDP) instead of DDP.

```
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.4210
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.3568
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.4056
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.2974
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.0243
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.0704
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.0503
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.0400
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.2057
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.0352
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.0217
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.0306
Epoch: 0001/0003 | Train acc.: 93.65% | Val acc.: 95.74%
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0102
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0124
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0907
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0597
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0201
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0815
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0039
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0473
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.1221
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.0371
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.1342
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.0045
Epoch: 0002/0003 | Train acc.: 97.72% | Val acc.: 96.36%
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0131
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0122
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0182
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.1417
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0007
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0154
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0010
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0020
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.0134
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.4331
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.0044
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.2052
Epoch: 0003/0003 | Train acc.: 98.64% | Val acc.: 96.28%
Time elapsed 2.23 min
Memory used: 2.83 GB
Test accuracy 96.22%
```



## 07_fabric-vit-mixed-fsdp-with-scheduler.py

Like `06_fabric-vit-mixed-fsdp.py` above but using a learning rate scheduler.



```
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.2974
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.3568
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.4056
Epoch: 0001/0003 | Batch 0000/0703 | Loss: 2.4210
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.0243
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.0503
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.0400
Epoch: 0001/0003 | Batch 0300/0703 | Loss: 0.0704
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.2057
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.0352
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.0217
Epoch: 0001/0003 | Batch 0600/0703 | Loss: 0.0306
Epoch: 0001/0003 | Train acc.: 93.65% | Val acc.: 95.74%
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0124
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0102
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0907
Epoch: 0002/0003 | Batch 0000/0703 | Loss: 0.0597
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0815
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0201
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0039
Epoch: 0002/0003 | Batch 0300/0703 | Loss: 0.0473
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.0371
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.1221
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.1342
Epoch: 0002/0003 | Batch 0600/0703 | Loss: 0.0045
Epoch: 0002/0003 | Train acc.: 97.72% | Val acc.: 96.36%
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0131
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0122
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.0182
Epoch: 0003/0003 | Batch 0000/0703 | Loss: 0.1417
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0021
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0317
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0491
Epoch: 0003/0003 | Batch 0300/0703 | Loss: 0.0174
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.0013
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.0062
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.0013
Epoch: 0003/0003 | Batch 0600/0703 | Loss: 0.0303
Epoch: 0003/0003 | Train acc.: 98.66% | Val acc.: 95.60%
Time elapsed 2.26 min
Memory used: 2.83 GB
Test accuracy 96.43%
```

