import re
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model
import gradio as gr
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Tesent.csv"
TEACHER_PATH = ROOT / "best_muril_lora.pt"
TEACHER_PROBS_PATH = ROOT / "teacher_probs.pt"
DISTIL_PATH = ROOT / "best_student.pt"
BILSTM_PATH = ROOT / "best_bilstm_offline_kd.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
NUM_CLASSES = 3
LABEL_TO_ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def basic_text_normalize(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Content": "text", "Label": "label"})
    df = df[["text", "label"]].drop_duplicates().dropna().reset_index(drop=True)
    df["text"] = df["text"].apply(basic_text_normalize)
    df["label"] = df["label"].map(LABEL_TO_ID)
    df = df.dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


class IndexedKDDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=max_len,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["idx"] = idx
        return item


class TeacherInferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            dataframe["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_len,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


class MurilPEFTModel(nn.Module):
    def __init__(self, base_model, num_classes=NUM_CLASSES):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def mean_pool(self, output, mask):
        token_embeddings = output.last_hidden_state
        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(output, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return logits


class BiLSTMStudent(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def mean_pool(self, outputs, mask):
        mask = mask.unsqueeze(-1)
        summed = torch.sum(outputs * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        outputs, _ = self.lstm(x)
        pooled = self.mean_pool(outputs, attention_mask)
        logits = self.fc(self.dropout(pooled))
        return logits


def build_teacher():
    model_name = "google/muril-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    base_model = get_peft_model(base_model, lora_config)
    model = MurilPEFTModel(base_model).to(DEVICE)
    return tokenizer, model


def evaluate_teacher(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(1, len(loader)), correct / max(1, total)


def train_teacher(df):
    tokenizer, model = build_teacher()
    if TEACHER_PATH.exists():
        print(f"Teacher checkpoint found: {TEACHER_PATH.name}")
        model.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
        return tokenizer, model

    df_train, df_temp = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    df_val, _ = train_test_split(
        df_temp, test_size=0.5, stratify=df_temp["label"], random_state=42
    )

    train_loader = DataLoader(SentimentDataset(df_train, tokenizer), batch_size=32, shuffle=True)
    val_loader = DataLoader(SentimentDataset(df_val, tokenizer), batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    epochs = 6
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * len(train_loader) * epochs),
        num_training_steps=len(train_loader) * epochs,
    )

    best_val_acc = 0.0
    patience = 2
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        start = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_loss, val_acc = evaluate_teacher(model, val_loader, criterion)
        print(
            f"Teacher epoch {epoch + 1}/{epochs} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"val_loss={val_loss:.4f} | elapsed={(time.time() - start) / 60:.1f}m"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), TEACHER_PATH)
            print(f"Saved {TEACHER_PATH.name}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Teacher early stopping triggered")
                break

    model.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
    return tokenizer, model


def generate_teacher_probs(df, tokenizer, teacher_model):
    if TEACHER_PROBS_PATH.exists():
        print(f"Teacher probabilities found: {TEACHER_PROBS_PATH.name}")
        return torch.load(TEACHER_PROBS_PATH, map_location="cpu").float()

    teacher_model.eval()
    teacher_loader = DataLoader(
        TeacherInferenceDataset(df, tokenizer), batch_size=32, shuffle=False
    )
    all_teacher_logits = []
    with torch.no_grad():
        for batch in teacher_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = teacher_model(input_ids, attention_mask)
            all_teacher_logits.append(logits.cpu())
    teacher_logits = torch.cat(all_teacher_logits, dim=0)
    teacher_probs = torch.softmax(teacher_logits, dim=1)
    torch.save(teacher_probs, TEACHER_PROBS_PATH)
    print(f"Saved {TEACHER_PROBS_PATH.name}")
    return teacher_probs


def kd_loss(student_logits, teacher_values, labels, alpha=0.7, temperature=2.0, teacher_are_probs=False):
    hard_loss = F.cross_entropy(student_logits, labels)
    teacher_dist = teacher_values
    if not teacher_are_probs:
        teacher_dist = F.softmax(teacher_values / temperature, dim=1)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        teacher_dist,
        reduction="batchmean",
    ) * (temperature * temperature)
    return alpha * hard_loss + (1 - alpha) * soft_loss


def train_distilbert_student(df, teacher_probs):
    student_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    student = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased",
        num_labels=NUM_CLASSES,
    ).to(DEVICE)

    if DISTIL_PATH.exists():
        print(f"DistilBERT checkpoint found: {DISTIL_PATH.name}")
        student.load_state_dict(torch.load(DISTIL_PATH, map_location=DEVICE))
        return student_tokenizer, student

    dataset = IndexedKDDataset(df, student_tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-5)
    epochs = 5

    student.train()
    for epoch in range(epochs):
        total_loss = 0.0
        start = time.time()
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            idx = batch["idx"]

            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = outputs.logits
            batch_teacher_probs = teacher_probs[idx].to(DEVICE)
            loss = kd_loss(
                student_logits,
                batch_teacher_probs,
                labels,
                alpha=0.7,
                temperature=2.0,
                teacher_are_probs=True,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(
            f"DistilBERT epoch {epoch + 1}/{epochs} | "
            f"avg_loss={avg_loss:.4f} | elapsed={(time.time() - start) / 60:.1f}m"
        )

    torch.save(student.state_dict(), DISTIL_PATH)
    print(f"Saved {DISTIL_PATH.name}")
    student.eval()
    return student_tokenizer, student


def train_bilstm_student(df, teacher_model, muril_tokenizer):
    student = BiLSTMStudent(vocab_size=muril_tokenizer.vocab_size).to(DEVICE)
    if BILSTM_PATH.exists():
        print(f"BiLSTM checkpoint found: {BILSTM_PATH.name}")
        student.load_state_dict(torch.load(BILSTM_PATH, map_location=DEVICE))
        return student

    df_train, df_temp = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    df_val, _ = train_test_split(
        df_temp, test_size=0.5, stratify=df_temp["label"], random_state=42
    )
    train_loader = DataLoader(SentimentDataset(df_train, muril_tokenizer), batch_size=32, shuffle=True)
    val_loader = DataLoader(SentimentDataset(df_val, muril_tokenizer), batch_size=32, shuffle=False)

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    best_val_acc = 0.0
    patience = 2
    no_improve = 0
    epochs = 8

    def evaluate_student(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / max(1, total)

    for epoch in range(epochs):
        student.train()
        start = time.time()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask)
            student_logits = student(input_ids, attention_mask)
            loss = kd_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=3.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        val_acc = evaluate_student(student, val_loader)
        print(
            f"BiLSTM epoch {epoch + 1}/{epochs} | "
            f"val_acc={val_acc:.4f} | elapsed={(time.time() - start) / 60:.1f}m"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(student.state_dict(), BILSTM_PATH)
            print(f"Saved {BILSTM_PATH.name}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("BiLSTM early stopping triggered")
                break

    student.load_state_dict(torch.load(BILSTM_PATH, map_location=DEVICE))
    student.eval()
    return student


def encode_text(text, tokenizer):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return {k: v.to(DEVICE) for k, v in enc.items() if k != "token_type_ids"}


def probs_to_dict(probs):
    return {
        "Negative": float(probs[0]),
        "Neutral": float(probs[1]),
        "Positive": float(probs[2]),
    }


def trust_decision(pred_label, confidence, threshold=0.6):
    if confidence < threshold:
        return "UNCERTAIN"
    return pred_label


def detect_failure(text, confidence):
    if len(text.strip()) < 5:
        return "Input too short"
    if confidence < 0.5:
        return "Low confidence"
    return "OK"


def get_token_importance(text, model_fn):
    words = text.split()[:6]
    if len(words) <= 1:
        return []
    base_probs, _ = model_fn(text)
    base_conf = float(max(base_probs))
    scores = []
    for i in range(len(words)):
        new_text = " ".join(words[:i] + words[i + 1 :])
        new_probs, _ = model_fn(new_text)
        impact = base_conf - float(max(new_probs))
        scores.append((words[i], impact))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def format_explanation(importances):
    if not importances:
        return "No explanation"
    text = "### Important Words\n\n"
    for word, score in importances:
        text += f"- **{word}** ({score:.3f})\n"
    return text


def make_prob_chart(prob_dict, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(prob_dict.keys()), list(prob_dict.values()))
    ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def make_comparison_chart(t, d, b):
    labels = ["Negative", "Neutral", "Positive"]
    x = list(range(3))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width for i in x], t, width, label="Teacher")
    ax.bar(x, d, width, label="DistilBERT")
    ax.bar([i + width for i in x], b, width, label="BiLSTM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    return fig


def format_single_result(model_name, probs, elapsed, text, model_fn):
    pred_idx = int(probs.argmax())
    pred_label = ID_TO_LABEL[pred_idx]
    confidence = float(probs[pred_idx])
    final_label = trust_decision(pred_label, confidence)
    failure_msg = detect_failure(text, confidence)
    explanation = format_explanation(get_token_importance(text, model_fn))
    return (
        f"**Model:** {model_name}\n\n"
        f"**Prediction:** {final_label}\n\n"
        f"**Confidence:** {confidence:.4f}\n\n"
        f"**Latency:** {elapsed:.2f} ms\n\n"
        f"**Status:** {failure_msg}\n\n"
        f"{explanation}"
    )


def build_demo(muril_tokenizer, student_tokenizer, teacher_model, distil_model, bilstm_model):
    @torch.no_grad()
    def predict_teacher(text):
        inputs = encode_text(text, muril_tokenizer)
        start = time.time()
        logits = teacher_model(**inputs)
        elapsed = (time.time() - start) * 1000
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return probs, elapsed

    @torch.no_grad()
    def predict_distilbert(text):
        inputs = encode_text(text, student_tokenizer)
        start = time.time()
        outputs = distil_model(**inputs)
        elapsed = (time.time() - start) * 1000
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        return probs, elapsed

    @torch.no_grad()
    def predict_bilstm(text):
        inputs = encode_text(text, muril_tokenizer)
        start = time.time()
        logits = bilstm_model(inputs["input_ids"], inputs["attention_mask"])
        elapsed = (time.time() - start) * 1000
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return probs, elapsed

    def run_selected_model(text, model_name, language):
        if model_name == "Teacher (MuRIL + LoRA)":
            probs, elapsed = predict_teacher(text)
            model_fn = predict_teacher
        elif model_name == "Student (DistilBERT)":
            probs, elapsed = predict_distilbert(text)
            model_fn = predict_distilbert
        else:
            probs, elapsed = predict_bilstm(text)
            model_fn = predict_bilstm

        result = format_single_result(model_name, probs, elapsed, text, model_fn)
        fig = make_prob_chart(probs_to_dict(probs), f"{model_name} Probabilities")
        return result, fig

    def compare_all_models(text, language):
        tp, t1 = predict_teacher(text)
        dp, t2 = predict_distilbert(text)
        bp, t3 = predict_bilstm(text)
        rows = []
        for name, probs, latency in [
            ("Teacher (MuRIL + LoRA)", tp, t1),
            ("Student (DistilBERT)", dp, t2),
            ("Student (BiLSTM)", bp, t3),
        ]:
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
            rows.append(
                {
                    "Model": name,
                    "Prediction": trust_decision(ID_TO_LABEL[pred_idx], confidence),
                    "Confidence": round(confidence, 4),
                    "Negative": round(float(probs[0]), 4),
                    "Neutral": round(float(probs[1]), 4),
                    "Positive": round(float(probs[2]), 4),
                    "Latency (ms)": round(float(latency), 2),
                }
            )
        df = pd.DataFrame(rows)
        fig = make_comparison_chart(tp, dp, bp)
        return df, fig

    examples = [
        ["ఈ సినిమా చాలా బాగుంది", "Teacher (MuRIL + LoRA)", "Telugu"],
        ["ಈ ಸಿನಿಮಾ ಚೆನ್ನಾಗಿದೆ", "Student (DistilBERT)", "Kannada"],
        ["हा चित्रपट खूप छान आहे", "Student (BiLSTM)", "Marathi"],
    ]

    with gr.Blocks(title="Trustworthy Multilingual Sentiment AI") as demo:
        gr.Markdown("# Trustworthy Multilingual Sentiment AI")
        gr.Markdown("MuRIL + LoRA teacher with DistilBERT and BiLSTM student models")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="Enter text", lines=4)
                model_choice = gr.Dropdown(
                    ["Teacher (MuRIL + LoRA)", "Student (DistilBERT)", "Student (BiLSTM)"],
                    value="Teacher (MuRIL + LoRA)",
                )
                language_choice = gr.Dropdown(
                    ["Telugu", "Kannada", "Marathi"],
                    value="Telugu",
                )
                predict_btn = gr.Button("Predict")
                compare_btn = gr.Button("Compare All Models")

            with gr.Column():
                result_output = gr.Markdown()
                chart_output = gr.Plot()

        comparison_table = gr.Dataframe()
        comparison_chart = gr.Plot()

        gr.Examples(examples=examples, inputs=[text_input, model_choice, language_choice])

        predict_btn.click(
            fn=run_selected_model,
            inputs=[text_input, model_choice, language_choice],
            outputs=[result_output, chart_output],
        )

        compare_btn.click(
            fn=compare_all_models,
            inputs=[text_input, language_choice],
            outputs=[comparison_table, comparison_chart],
        )

    return demo


def main():
    print(f"Using device: {DEVICE}")
    df = load_dataset()
    print(f"Loaded dataset with {len(df)} rows")

    muril_tokenizer, teacher_model = train_teacher(df)
    teacher_model.eval()

    teacher_probs = generate_teacher_probs(df, muril_tokenizer, teacher_model)
    student_tokenizer, distil_model = train_distilbert_student(df, teacher_probs)
    distil_model.eval()

    bilstm_model = train_bilstm_student(df, teacher_model, muril_tokenizer)
    bilstm_model.eval()

    demo = build_demo(
        muril_tokenizer,
        student_tokenizer,
        teacher_model,
        distil_model,
        bilstm_model,
    )
    print("Launching app at http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
