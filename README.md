# Performance testing of the bgmv/sgmv(triton) ops

## 1. Motivation

To fully compare the efficiency differences between the Triton kernel we implemented([#5036](https://github.com/vllm-project/vllm/pull/5036))and the kernel implemented by FurtherAI([#5336](https://github.com/vllm-project/vllm/pull/5356))in various scenarios, we conducted performance testing of the operator.





## 2. Description

### 2.1 Requirements

- vllm == 0.4.3
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
- We didn't have much timey, so we couldn't perform benchmarking like `bgmv`. However, sgmv's benchmarking can be done in the same way as `bgmv`.
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

- If you have any questions, please feel free to contact me.

## 4 TODO 

-  ~~Performance testing of the SGMV kernel~~