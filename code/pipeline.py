from typing import List
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)
import torch
from datasets import Dataset


class QuestionBankService:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model, self.tokenizer = self._load_model_and_tokenizer()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model or tokenizer: {e}")

    def _load_model_and_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            model.resize_token_embeddings(len(tokenizer))
            model.to(self.device)
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Error loading model and tokenizer: {e}")

    def _build_prompt(self, question: str, options: str, correct_answer: str, explanation: str) -> str:
        try:
            return f"""
            Original Question: {question}
            Options: {options}
            Correct Answer: {correct_answer}
            Explanation: {explanation}

            Generate 4 distinct questions based on the original question. For each question:
            - Provide 4 options (one correct, three incorrect).
            - Clearly label the correct answer and provide an explanation.
            Output format:
            1. Question: ...
            Options: a) ... b) ... c) ... d) ...
            Correct Answer: ...
            Explanation: ...
            """
        except Exception as e:
            raise ValueError(f"Error building prompt: {e}")

    def generate_distinct_questions(self, prompt: str) -> List[str]:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=512,
                    num_return_sequences=4,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        except Exception as e:
            raise RuntimeError(f"Error generating distinct questions: {e}")

    def parse_generated_output(self, generated_text: str) -> List[dict]:
        try:
            questions = []
            blocks = generated_text.split("1. Question:")[1:]
            for block in blocks:
                lines = block.strip().split("\n")
                question = lines[0].replace("Question:", "").strip()
                options_line = next((line for line in lines if line.startswith("Options:")), None)
                correct_answer_line = next((line for line in lines if line.startswith("Correct Answer:")), None)
                explanation_line = next((line for line in lines if line.startswith("Explanation:")), None)

                if options_line and correct_answer_line and explanation_line:
                    options = options_line.replace("Options:", "").strip()
                    correct_answer = correct_answer_line.replace("Correct Answer:", "").strip()
                    explanation = explanation_line.replace("Explanation:", "").strip()

                    questions.append({
                        "Generated Question": question,
                        "Options": options,
                        "Correct Answer": correct_answer,
                        "Explanation": explanation,
                    })
            return questions
        except Exception as e:
            raise RuntimeError(f"Error parsing generated output: {e}")

    def expand_questions(self, input_data: pd.DataFrame) -> pd.DataFrame:
        try:
            required_columns = ["Degree", "Role", "Section", "Proficiency Level", "Question", "Options", "Correct Answer", "Explanation"]
            for column in required_columns:
                if column not in input_data.columns:
                    raise ValueError(f"Missing required column: {column}")

            expanded_questions = []
            for _, row in input_data.iterrows():
                try:
                    prompt = self._build_prompt(
                        row["Question"], row["Options"], row["Correct Answer"], row["Explanation"]
                    )
                    generated_texts = self.generate_distinct_questions(prompt)
                    for generated_text in generated_texts:
                        parsed_questions = self.parse_generated_output(generated_text)
                        for parsed_question in parsed_questions:
                            expanded_questions.append({
                                "Degree": row["Degree"],
                                "Role": row["Role"],
                                "Section": row["Section"],
                                "Proficiency Level": row["Proficiency Level"],
                                **parsed_question,
                            })
                except Exception as e:
                    print(f"Error processing row {row.to_dict()}: {e}")

            return pd.DataFrame(expanded_questions)
        except Exception as e:
            raise RuntimeError(f"Error expanding questions: {e}")

    def preprocess_dataset(self, dataset: Dataset, max_length: int = 512):
        try:
            def tokenize_function(examples):
                inputs = self.tokenizer(
                    examples["Question"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )
                targets = self.tokenizer(
                    examples["Correct Answer"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )
                inputs["labels"] = targets["input_ids"]
                return inputs

            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            return tokenized_dataset.train_test_split(test_size=0.2)
        except Exception as e:
            raise RuntimeError(f"Error preprocessing dataset: {e}")

    def fine_tune_model(self, input_data: pd.DataFrame, output_dir: str):
        try:
            dataset = Dataset.from_pandas(input_data)
            tokenized_dataset = self.preprocess_dataset(dataset)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=10,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                eval_strategy="epoch",
                fp16=False,
            )

            trainer = Trainer(
                model=self.model,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                args=training_args,
            )

            trainer.train()
            trainer.save_model(output_dir)
            print(f"Model fine-tuned and saved to {output_dir}")
        except Exception as e:
            raise RuntimeError(f"Error fine-tuning model: {e}")

if __name__ == "__main__":
    import os
    try:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        file_path = "placement-questions-excel.csv"
        input_data = pd.read_csv(file_path, encoding="ISO-8859-1")

        service = QuestionBankService()
        fine_tuned_model_dir = "fine_tuned_gpt2_model"
        service.fine_tune_model(input_data, output_dir=fine_tuned_model_dir)

        expanded_questions = service.expand_questions(input_data)
        expanded_questions.to_csv("expanded_questions.csv", index=False)
        print("Expanded questions saved to expanded_questions.csv")
    except Exception as e:
        print(f"Error in main execution: {e}")
