# Performance testing of the bgmv/sgmv(triton) ops

## 1. Motivation




## 2. Description

### 2.1 Requirements

- vllm == 0.5.3
- torch == 2.3.0
- triton == 2.3.0

### 2.2 Testing(bgmv)

#### Runing benchmark script
We provide a benchmark script to generate the optimal kernel configuration. Please execute this script before testing to obtain the optimal kernel configuration.

Run it in the current directory:

```bash
python get_bgmv_configs.py
```


After executing the benchmark, you can find the configuration files in `jee_ops/bgmv_config`.


PS:

- This typically takes several hours to complete. If the benchmark is not executed, our kernel will use the default configuration. Even with the default configuration, our kernel still demonstrates advantages in the testing(RTX 3090), please refer to: `bgmv_results_rtx3090/`(results_rtx3090)



#### Runing test script

Then execute the following script to start the testing:

```bash
python test_punica_sgmv_speed.py
```
- the result saved in the  current directory.


####  Result

We have already conducted testing on an RTX 3090 GPU and obtained results for both `default_config_result` and `tuned_config_result` configurations. You can find these test results in directory `bgmv_results_rtx3090/`.


### 2.3 Testing(sgmv)
- We couldn't perform benchmarking like `bgmv`(All use the default configuration). However, sgmv's benchmarking can be done in the same way as `bgmv`. 
#### Runing test script

Then execute the following script to start the testing:

```bash
python test_punica_sgmv_speed.py
```
- the result saved in the  current directory.


####  Result

We have already conducted testing on an RTX 3090 GPU and obtained results for both `seq_len=128` and `seq_len=512` . You can find these test results in directory `sgmv_results_rtx3090/`.

## 3. Explanation

- In our testing, we utilize `cuda graph` to  reduce the runtime overhead of the triton. This approach allows our testing results to closely approximate the actual execution time of the  on the device.

