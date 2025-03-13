import gpn.model
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 加载模型
model_name = "songlab/gpn-brassicales"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

# 创建输入数据
inputs = tokenizer("ATCGTGA[MASK]AACG", return_tensors="pt")

# 获取模型输出
outputs = model(**inputs)

# 打印 logits 结果
print(outputs.logits)
