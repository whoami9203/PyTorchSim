# 指令集
設置環境變數,輸出位置
```bash
#source setup_env_var.sh ${TOGSIM_SSD_TRACE_NAME} [0/1] [0/1] [TOGSIM_CONFIG_PATH]
# Original
source setup_env_var.sh test1 0 0 configs/eclab_cambricon_llama2_7B.yml
# With DRAM simlet linked by LegoSim (NoC always on when this or SSD is on)
source setup_env_var.sh test1 0 1 configs/eclab_cambricon_llama2_7B.yml
# With SSD simlet linked by LegoSim (NoC always on when this or SSD is on)
source setup_env_var.sh test1 1 0 configs/eclab_cambricon_llama2_7B.yml
```

Llama2-7B 存取權
```bash
1. 去 meta-llama/Llama-2-7b-hf 頁面，確認有填表申請並且 Meta 已批准
2. 到 huggingface.co/settings/tokens 建立一個 token（Read 權限就夠）
3. 在 container 裡登入
   huggingface-cli login
   貼上 token 後 Enter
```

清除輸出
```bash
# 保留cache
bash cleanup_results.sh 0
# 清除cache (會重新跑gem5 建議清除)
bash cleanup_results.sh 1
``` 

執行模擬
```bash
# 官方範例
python3 tests/Llama/test_llama.py

# sim開頭: 跑random input
# TinyLLaMA
#python3 test/Llama/test_tinyllama.py --npu --phase [decode/prefill] --num_layers [i] --seq_len [default: 500 (for prefill)] --context_len [default: 500 (for decode)] --dtype [default: float32] --ssd-backend [formula/simplessd]
# setup_env_var.sh 有設SSD simlet為1 --ssd-backend 才有實際作用
python3 tests/Llama/sim_tinyllama.py --npu --phase decode --num_layers 1 --dtype float16

# LLaMA2-7B
python3 tests/Llama/sim_llama2_7B.py --npu --phase decode --num_layers 1 --dtype float16

# GPT_NeoX-20B
python3 tests/GPT/sim_GPT_NeoX_20B.py --npu --phase decode --num_layers 1 --dtype float16

# test開頭: 跑實際prompt (沒開functional mode效果跟sim開頭一樣)
#python3 test/Llama/test_tinyllama.py --npu --prompt [ex. Machine learning is a useful tool that] --max_new_tokens [default: 1]
python3 tests/Llama/test_tinyllama.py --npu 

# LLaMA2-7B
python3 tests/Llama/test_llama2_7B.py --npu

# GPT_NeoX-20B
python3 tests/GPT/test_GPT_NeoX_20B.py --npu
``` 

輸出總cycle數
```bash
#python3 sum_togsim_cycles.py --trace-name [ex. test1 (default: ${TOGSIM_SSD_TRACE_NAME})]
python3 sum_togsim_cycles.py
```

擷取model weights traces (先不用管)
```bash
python3 merge_weight_ranges.py
python3 extract_weight_traces.py
```

# 輸出位置
log檔案: togsim_results/${TOGSIM_SSD_TRACE_NAME}/

DMA traces: ssd_traces/${TOGSIM_SSD_TRACE_NAME}/