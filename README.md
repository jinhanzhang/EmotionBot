# EmotionBot
To finetune, run:
<pre>
python unsloth_finetune.py --max_new_tokens=64 --model=unsloth/Meta-Llama-3.1-8B-bnb-4bit --tokenizer=llama-3.1 --per_device_train_batch_size=2 --gradient_accumulation_steps=4 --single_turn=true
</pre>

To evaluate a model, run:
<pre>
python unsloth_eval.py --max_new_tokens=64 --model=unsloth/Meta-Llama-3.1-8B-bnb-4bit --tokenizer=llama-3.1 --per_device_train_batch_size=2 --gradient_accumulation_steps=4 --single_turn=true
</pre>

Results in outputs