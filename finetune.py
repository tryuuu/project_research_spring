import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
import datasets

model_name = "cyberagent/open-calm-1b"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-1b")

train_losses = []



class InstructDataset(Dataset):
    def __init__(self, json_file, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.features = []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
            
        for j in json_list:
            if 'input' in j:
                source_text = "しりとりをしましょう。つまり、次の言葉をひらがなに直した時、最後の文字から始まる言葉を一つだけ言ってください。" + j['input']
            else:
                source_text = "しりとりをしましょう。つまり、次の言葉をひらがなに直した時、最後の文字から始まる言葉を一つだけ言ってください。"
            
            example_text = source_text + j['output']
            
            source_tokenized = self.tokenizer(
                source_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_length=True,
                return_tensors='pt'
            )
            
            example_tokenized = self.tokenizer(
                example_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = example_tokenized['input_ids'][0]
            labels = input_ids.clone()
            
            source_len = source_tokenized['length'][0]
            labels[:source_len] = self.ignore_index
            
            self.features.append({
                'input_ids': input_ids,
                'labels': labels
            })
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

train_dataset = InstructDataset("shiritori_data.json", tokenizer)

class InstructCollator():
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, examples):
        input_batch = []
        label_batch = []
        for example in examples:
            input_batch.append(example['input_ids'])
            label_batch.append(example['labels'])

        input_ids = pad_sequence(
            input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # labelsのpaddingトークンは先程と同様にignore_indexである-100で埋める
        labels = pad_sequence(
            label_batch, batch_first=True, padding_value=self.ignore_index
        )

        # attention_maskはbool値でもいいらしい
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

collator = InstructCollator(tokenizer)
loader = DataLoader(train_dataset, collate_fn=collator, batch_size=8, shuffle=True)
batch = next(iter(loader))

"""for param in model.parameters():
    param.requires_grad = False # モデルをフリーズ
    if param.ndim == 1:
        # 安定のためにレイヤーノルムをfp32にキャスト
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.embed_out = CastOutputToFloat(model.embed_out)

from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.CAUSAL_LM
)"""

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
        output_dir='./output',
        save_total_limit=1,
        per_device_train_batch_size=8,
        num_train_epochs=16,#8 #7bの場合、8<9<10 
        remove_unused_columns=False,
        logging_steps=20,
        fp16=True,
        dataloader_num_workers=16,
        report_to="none",
        learning_rate=4e-5#4e-4
)

trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset,
)

model.config.use_cache = False
trainer.train()

n=0
while(n<100):
    model = trainer.model
    instruction = "しりとりをしましょう。つまり、次の言葉をひらがなに直した時、最後の文字から始まる言葉を一つだけ言ってください。"
    input_text = input("入力して: ")
    input_ids = tokenizer.encode(instruction, input_text, add_special_tokens=True, truncation=True, padding="longest", return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
        input_ids=input_ids,
        max_length=24,  # 生成するテキストの最大長さ
        num_return_sequences=1,  # 生成するテキストの数
        temperature=0.7,  # ランダム性を制御する温度パラメータ
        )
    # IDを元の日本語にデコード
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    output = output.replace(instruction, '').replace(input_text, '').strip()

    print(output)  
    n+=1

