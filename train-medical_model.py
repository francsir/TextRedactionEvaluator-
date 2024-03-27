from transformers import AutoTokenizer, AutoModelForMaskedLM, default_data_collator, TrainingArguments, Trainer, get_scheduler
import collections
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
import torch
import math
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm



model_checkpoint = "medicalai/ClinicalBert"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#imdb_dataset = load_dataset("imdb")
medical = Dataset.from_csv('medicalData\Medical-Abstracts-TC-Corpus\medical_tc_train.csv')
chunk_size = 128
wwm_probability = 0.2


def tokenize_function(examples):
    result = tokenizer(examples["medical_abstract"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

def downsample_dataset(dataset, train_size):
    test_size = int(0.1 * train_size)

    downsampled_dataset = dataset.train_test_split(
        train_size=train_size, test_size=test_size, seed = 42
        )
    return downsampled_dataset

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = whole_word_masking_data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

tokenized_datasets = medical.map(
    tokenize_function, batched=True, remove_columns=["condition_label", "medical_abstract"]
)

print(tokenized_datasets)
lm_datasets = tokenized_datasets.map(group_texts, batched=True)#

#samples = [lm_datasets["train"][i] for i in range(5)]
#batch = whole_word_masking_data_collator(samples)

downsampled = downsample_dataset(lm_datasets, 10_000)

eval_dataset = downsampled["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

batch_size = 64


train_dataloader = DataLoader(
    downsampled["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=whole_word_masking_data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

optimizer = AdamW(model.parameters(), lr=5e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch


lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

output_dir = "model/clinical-trained"
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
    if accelerator.is_main_process:
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()

