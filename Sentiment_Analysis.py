{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "from transformers import AutoTokenizer, AutoModel\n",
    "from peft import LoraConfig, get_peft_model\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(\"Device:\", device)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "tesent_raw_df = pd.read_csv(\"Tesent.csv\")\n",
    "# adjust path if needed\n",
    "\n",
    "print(\"Shape:\", tesent_raw_df.shape)\n",
    "tesent_raw_df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "tesent_raw_df.info()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "tesent_explore_df = tesent_raw_df.copy()\n",
    "\n",
    "tesent_explore_df = tesent_explore_df.rename(\n",
    "    columns={\n",
    "        \"Content\": \"text\",\n",
    "        \"Label\": \"label\"\n",
    "    }\n",
    ")\n",
    "\n",
    "tesent_explore_df[[\"text\", \"label\"]].head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "tesent_explore_df[\"label\"].value_counts()\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "tesent_clean_df = tesent_explore_df.copy()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "tesent_clean_df = tesent_clean_df.drop_duplicates(subset=[\"text\", \"label\"])\n",
    "tesent_clean_df.reset_index(drop=True, inplace=True)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "tesent_clean_df[\"text\"].str.len() > 3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import re\n",
    "\n",
    "def basic_text_normalize(text):\n",
    "    text = str(text)\n",
    "    text = re.sub(r\"http\\S+|www\\S+\", \"\", text)\n",
    "    text = re.sub(r\"\\s+\", \" \", text).strip()\n",
    "    return text\n",
    "\n",
    "tesent_clean_df[\"text\"] = tesent_clean_df[\"text\"].apply(basic_text_normalize)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "label_map = {\n",
    "    \"Negative\": 0,\n",
    "    \"Neutral\": 1,\n",
    "    \"Positive\": 2\n",
    "}\n",
    "\n",
    "tesent_clean_df[\"label\"] = tesent_clean_df[\"label\"].map(label_map)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "!pip install transformers peft accelerate -q\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "from transformers import AutoTokenizer, AutoModel\n",
    "from transformers import get_linear_schedule_with_warmup\n",
    "\n",
    "from peft import LoraConfig, get_peft_model, TaskType\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "df = tesent_clean_df.copy()\n",
    "\n",
    "df = df[[\"text\", \"label\"]]   # keep only what we need\n",
    "df.dropna(inplace=True)\n",
    "\n",
    "assert set(df[\"label\"].unique()) <= {0,1,2}\n",
    "\n",
    "df.reset_index(drop=True, inplace=True)\n",
    "\n",
    "print(\"Dataset size:\", len(df))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "df_train, df_temp = train_test_split(\n",
    "    df,\n",
    "    test_size=0.2,\n",
    "    stratify=df[\"label\"],\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "df_val, df_test = train_test_split(\n",
    "    df_temp,\n",
    "    test_size=0.5,\n",
    "    stratify=df_temp[\"label\"],\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "for name, data in zip(\n",
    "    [\"Train\",\"Val\",\"Test\"],\n",
    "    [df_train, df_val, df_test]\n",
    "):\n",
    "    print(name)\n",
    "    print(data[\"label\"].value_counts(normalize=True))\n",
    "    print()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "model_name = \"google/muril-base-cased\"\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "class SentimentDataset(Dataset):\n",
    "    def __init__(self, df, tokenizer, max_len=128):\n",
    "        self.texts = df[\"text\"].tolist()\n",
    "        self.labels = df[\"label\"].tolist()\n",
    "        self.tokenizer = tokenizer\n",
    "        self.max_len = max_len\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.texts)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        encoding = self.tokenizer(\n",
    "            self.texts[idx],\n",
    "            padding=\"max_length\",\n",
    "            truncation=True,\n",
    "            max_length=self.max_len,\n",
    "            return_tensors=\"pt\"\n",
    "        )\n",
    "        \n",
    "        return {\n",
    "            \"input_ids\": encoding[\"input_ids\"].squeeze(),\n",
    "            \"attention_mask\": encoding[\"attention_mask\"].squeeze(),\n",
    "            \"label\": torch.tensor(self.labels[idx], dtype=torch.long)\n",
    "        }\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "batch_size = 32\n",
    "\n",
    "train_loader = DataLoader(\n",
    "    SentimentDataset(df_train, tokenizer),\n",
    "    batch_size=batch_size,\n",
    "    shuffle=True\n",
    ")\n",
    "\n",
    "val_loader = DataLoader(\n",
    "    SentimentDataset(df_val, tokenizer),\n",
    "    batch_size=batch_size,\n",
    "    shuffle=False\n",
    ")\n",
    "\n",
    "test_loader = DataLoader(\n",
    "    SentimentDataset(df_test, tokenizer),\n",
    "    batch_size=batch_size,\n",
    "    shuffle=False\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "base_model = AutoModel.from_pretrained(model_name)\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    target_modules=[\"query\",\"key\",\"value\",\"dense\"],  # strong coverage\n",
    "    lora_dropout=0.1,\n",
    "    bias=\"none\",\n",
    "    task_type=TaskType.FEATURE_EXTRACTION\n",
    ")\n",
    "\n",
    "base_model = get_peft_model(base_model, lora_config)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "class MurilPEFTModel(nn.Module):\n",
    "    def __init__(self, base_model, num_classes=3):\n",
    "        super().__init__()\n",
    "        self.base_model = base_model\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.classifier = nn.Linear(768, num_classes)\n",
    "        \n",
    "    def mean_pool(self, output, mask):\n",
    "        token_embeddings = output.last_hidden_state\n",
    "        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()\n",
    "        \n",
    "        summed = torch.sum(token_embeddings * mask, 1)\n",
    "        counts = torch.clamp(mask.sum(1), min=1e-9)\n",
    "        \n",
    "        return summed / counts\n",
    "    \n",
    "    def forward(self, input_ids, attention_mask):\n",
    "        output = self.base_model(\n",
    "            input_ids=input_ids,\n",
    "            attention_mask=attention_mask\n",
    "        )\n",
    "        \n",
    "        pooled = self.mean_pool(output, attention_mask)\n",
    "        logits = self.classifier(self.dropout(pooled))\n",
    "        \n",
    "        return logits\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "model = MurilPEFTModel(base_model).to(device)\n",
    "\n",
    "model.base_model.print_trainable_parameters()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "criterion = nn.CrossEntropyLoss()\n",
    "\n",
    "optimizer = torch.optim.AdamW(\n",
    "    model.parameters(),\n",
    "    lr=2e-4,              \n",
    "    weight_decay=0.01\n",
    ")\n",
    "\n",
    "epochs = 6\n",
    "\n",
    "scheduler = get_linear_schedule_with_warmup(\n",
    "    optimizer,\n",
    "    num_warmup_steps=int(0.1 * len(train_loader) * epochs),\n",
    "    num_training_steps=len(train_loader) * epochs\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def evaluate(model, loader):\n",
    "    model.eval()\n",
    "    \n",
    "    total_loss = 0\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        for batch in loader:\n",
    "            \n",
    "            input_ids = batch[\"input_ids\"].to(device)\n",
    "            attention_mask = batch[\"attention_mask\"].to(device)\n",
    "            labels = batch[\"label\"].to(device)\n",
    "            \n",
    "            logits = model(input_ids, attention_mask)\n",
    "            loss = criterion(logits, labels)\n",
    "            \n",
    "            total_loss += loss.item()\n",
    "            \n",
    "            preds = torch.argmax(logits, dim=1)\n",
    "            correct += (preds == labels).sum().item()\n",
    "            total += labels.size(0)\n",
    "            \n",
    "    return total_loss / len(loader), correct / total\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "best_val_acc = 0\n",
    "patience = 2\n",
    "no_improve = 0\n",
    "\n",
    "SAVE_PATH = \"best_muril_lora.pt\"\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    \n",
    "    model.train()\n",
    "    \n",
    "    total_loss = 0\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    \n",
    "    for batch in train_loader:\n",
    "        \n",
    "        input_ids = batch[\"input_ids\"].to(device)\n",
    "        attention_mask = batch[\"attention_mask\"].to(device)\n",
    "        labels = batch[\"label\"].to(device)\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        logits = model(input_ids, attention_mask)\n",
    "        loss = criterion(logits, labels)\n",
    "        \n",
    "        loss.backward()\n",
    "        \n",
    "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
    "        \n",
    "        optimizer.step()\n",
    "        scheduler.step()\n",
    "        \n",
    "        total_loss += loss.item()\n",
    "        \n",
    "        preds = torch.argmax(logits, dim=1)\n",
    "        correct += (preds == labels).sum().item()\n",
    "        total += labels.size(0)\n",
    "    \n",
    "    train_acc = correct / total\n",
    "    \n",
    "    val_loss, val_acc = evaluate(model, val_loader)\n",
    "    \n",
    "    print(f\"\\nEpoch {epoch+1}\")\n",
    "    print(\"Train Acc:\", train_acc)\n",
    "    print(\"Val Acc:\", val_acc)\n",
    "    \n",
    "    if val_acc > best_val_acc:\n",
    "        best_val_acc = val_acc\n",
    "        no_improve = 0\n",
    "        \n",
    "        torch.save(model.state_dict(), SAVE_PATH)\n",
    "        print(\"🔥 Best model saved!\")\n",
    "        \n",
    "    else:\n",
    "        no_improve += 1\n",
    "        \n",
    "    if no_improve >= patience:\n",
    "        print(\"Early stopping triggered.\")\n",
    "        break\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "model.load_state_dict(torch.load(SAVE_PATH))\n",
    "\n",
    "test_loss, test_acc = evaluate(model, test_loader)\n",
    "\n",
    "print(\"\\nFINAL TEST ACC:\", test_acc)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**STUDENT MODEL**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "teacher = MurilPEFTModel(base_model).to(device)\n",
    "\n",
    "teacher.load_state_dict(torch.load(SAVE_PATH))\n",
    "\n",
    "teacher.eval()\n",
    "\n",
    "for param in teacher.parameters():\n",
    "    param.requires_grad = False\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch.nn as nn\n",
    "\n",
    "class BiLSTMStudent(nn.Module):\n",
    "    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256, num_classes=3):\n",
    "        super().__init__()\n",
    "        \n",
    "        self.embedding = nn.Embedding(vocab_size, embed_dim)\n",
    "        \n",
    "        self.lstm = nn.LSTM(\n",
    "            embed_dim,\n",
    "            hidden_dim,\n",
    "            batch_first=True,\n",
    "            bidirectional=True\n",
    "        )\n",
    "        \n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.fc = nn.Linear(hidden_dim*2, num_classes)\n",
    "\n",
    "    def mean_pool(self, outputs, mask):\n",
    "        mask = mask.unsqueeze(-1)\n",
    "        summed = torch.sum(outputs * mask, dim=1)\n",
    "        counts = torch.clamp(mask.sum(dim=1), min=1e-9)\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "        \n",
    "        x = self.embedding(input_ids)\n",
    "        \n",
    "        outputs, _ = self.lstm(x)\n",
    "        \n",
    "        pooled = self.mean_pool(outputs, attention_mask)\n",
    "        \n",
    "        logits = self.fc(self.dropout(pooled))\n",
    "        \n",
    "        return logits\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "student = BiLSTMStudent(\n",
    "    vocab_size=tokenizer.vocab_size\n",
    ").to(device)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch.nn.functional as F\n",
    "\n",
    "def kd_loss(student_logits, teacher_logits, labels, alpha=0.5, T=3):\n",
    "\n",
    "    hard_loss = F.cross_entropy(student_logits, labels)\n",
    "\n",
    "    soft_loss = F.kl_div(\n",
    "        F.log_softmax(student_logits / T, dim=1),\n",
    "        F.softmax(teacher_logits / T, dim=1),\n",
    "        reduction='batchmean'\n",
    "    ) * (T*T)\n",
    "\n",
    "    return alpha * hard_loss + (1-alpha) * soft_loss\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "optimizer = torch.optim.AdamW(\n",
    "    student.parameters(),\n",
    "    lr=3e-4\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def evaluate_student(model, loader):\n",
    "    \n",
    "    model.eval()\n",
    "    \n",
    "    correct = 0\n",
    "    total = 0\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        for batch in loader:\n",
    "            \n",
    "            input_ids = batch[\"input_ids\"].to(device)\n",
    "            attention_mask = batch[\"attention_mask\"].to(device)\n",
    "            labels = batch[\"label\"].to(device)\n",
    "            \n",
    "            logits = model(input_ids, attention_mask)\n",
    "            \n",
    "            preds = torch.argmax(logits, dim=1)\n",
    "            \n",
    "            correct += (preds == labels).sum().item()\n",
    "            total += labels.size(0)\n",
    "            \n",
    "    return correct / total\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "best_val_acc = 0\n",
    "patience = 2\n",
    "no_improve = 0\n",
    "\n",
    "STUDENT_SAVE = \"bilstm_student.pt\"\n",
    "\n",
    "epochs = 8\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    \n",
    "    student.train()\n",
    "    \n",
    "    for batch in train_loader:\n",
    "        \n",
    "        input_ids = batch[\"input_ids\"].to(device)\n",
    "        attention_mask = batch[\"attention_mask\"].to(device)\n",
    "        labels = batch[\"label\"].to(device)\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        # 🔥 Teacher forward (NO GRAD)\n",
    "        with torch.no_grad():\n",
    "            teacher_logits = teacher(\n",
    "                input_ids,\n",
    "                attention_mask\n",
    "            )\n",
    "        \n",
    "        # Student forward\n",
    "        student_logits = student(\n",
    "            input_ids,\n",
    "            attention_mask\n",
    "        )\n",
    "        \n",
    "        loss = kd_loss(\n",
    "            student_logits,\n",
    "            teacher_logits,\n",
    "            labels\n",
    "        )\n",
    "        \n",
    "        loss.backward()\n",
    "        \n",
    "        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)\n",
    "        \n",
    "        optimizer.step()\n",
    "    \n",
    "    val_acc = evaluate_student(student, val_loader)\n",
    "    \n",
    "    print(f\"\\nEpoch {epoch+1}\")\n",
    "    print(\"Student Val Acc:\", val_acc)\n",
    "    \n",
    "    if val_acc > best_val_acc:\n",
    "        best_val_acc = val_acc\n",
    "        no_improve = 0\n",
    "        \n",
    "        torch.save(student.state_dict(), STUDENT_SAVE)\n",
    "        print(\"🔥 Best student saved!\")\n",
    "        \n",
    "    else:\n",
    "        no_improve += 1\n",
    "        \n",
    "    if no_improve >= patience:\n",
    "        print(\"Early stopping triggered.\")\n",
    "        break\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "student.load_state_dict(torch.load(STUDENT_SAVE))\n",
    "\n",
    "test_acc = evaluate_student(student, test_loader)\n",
    "\n",
    "print(\"\\nFINAL STUDENT TEST ACC:\", test_acc)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "teacher.eval()\n",
    "student.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "teacher.eval()\n",
    "student.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def compare_models(sentence):\n",
    "\n",
    "    inputs = tokenizer(\n",
    "        sentence,\n",
    "        return_tensors=\"pt\",\n",
    "        truncation=True,\n",
    "        padding=True,\n",
    "        max_length=128\n",
    "    ).to(device)\n",
    "\n",
    "    with torch.no_grad():\n",
    "\n",
    "        t_logits = teacher(\n",
    "            inputs[\"input_ids\"],\n",
    "            inputs[\"attention_mask\"]\n",
    "        )\n",
    "\n",
    "        s_logits = student(\n",
    "            inputs[\"input_ids\"],\n",
    "            inputs[\"attention_mask\"]\n",
    "        )\n",
    "\n",
    "    t_probs = torch.softmax(t_logits, dim=1)\n",
    "    s_probs = torch.softmax(s_logits, dim=1)\n",
    "\n",
    "    t_pred = t_probs.argmax().item()\n",
    "    s_pred = s_probs.argmax().item()\n",
    "\n",
    "    return (\n",
    "        t_pred,\n",
    "        t_probs.max().item(),\n",
    "        s_pred,\n",
    "        s_probs.max().item()\n",
    "    )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "label_map = {\n",
    "    0: \"Negative\",\n",
    "    1: \"Neutral\",\n",
    "    2: \"Positive\"\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "test_sentences = {\n",
    "\n",
    "    \"Telugu\": [\n",
    "        \"ఈ సినిమా చాలా బాగుంది\",        # very good\n",
    "        \"ఈ సినిమా చాలా చెత్తగా ఉంది\",  # very bad\n",
    "        \"ఈ సినిమా సరాసరి గా ఉంది\"      # average\n",
    "    ],\n",
    "\n",
    "    \"Kannada\": [\n",
    "        \"ಈ ಸಿನಿಮಾ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ\",\n",
    "        \"ಈ ಸಿನಿಮಾ ತುಂಬಾ ಕೆಟ್ಟಾಗಿದೆ\",\n",
    "        \"ಈ ಸಿನಿಮಾ ಸರಾಸರಿ ಇದೆ\"\n",
    "    ],\n",
    "\n",
    "    \"Tamil\": [\n",
    "        \"இந்த படம் மிகவும் நல்லது\",\n",
    "        \"இந்த படம் மிகவும் மோசமாக உள்ளது\",\n",
    "        \"இந்த படம் சராசரியாக உள்ளது\"\n",
    "    ]\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "for lang, sentences in test_sentences.items():\n",
    "\n",
    "    print(f\"\\n=========== {lang} ===========\")\n",
    "\n",
    "    for s in sentences:\n",
    "\n",
    "        t_pred, t_conf, s_pred, s_conf = compare_models(s)\n",
    "\n",
    "        print(f\"\\n{s}\")\n",
    "\n",
    "        print(f\"Teacher → {label_map[t_pred]} ({t_conf:.2f})\")\n",
    "        print(f\"Student → {label_map[s_pred]} ({s_conf:.2f})\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**DISTILLBERT**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "tesent_clean_df = tesent_clean_df.reset_index(drop=True)\n",
    "print(\"Dataset size:\", len(tesent_clean_df))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from transformers import AutoTokenizer\n",
    "from tqdm import tqdm\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(\"Using device:\", device)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "from transformers import AutoTokenizer\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "TEACHER_MODEL = \"google/muril-base-cased\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "teacher = MurilPEFTModel(base_model).to(device)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "SAVE_PATH = \"best_muril_lora.pt\"\n",
    "\n",
    "teacher.load_state_dict(\n",
    "    torch.load(SAVE_PATH, map_location=device)\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "teacher.eval()\n",
    "\n",
    "for p in teacher.parameters():\n",
    "    p.requires_grad = False\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "sample_text = \"ఈ సినిమా చాలా బాగుంది\"\n",
    "\n",
    "inputs = tokenizer(\n",
    "    sample_text,\n",
    "    return_tensors=\"pt\",\n",
    "    truncation=True,\n",
    "    padding=True\n",
    ").to(device)\n",
    "\n",
    "with torch.no_grad():\n",
    "    logits = teacher(inputs[\"input_ids\"], inputs[\"attention_mask\"])\n",
    "    probs = torch.softmax(logits, dim=1)\n",
    "\n",
    "print(probs)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from torch.utils.data import Dataset, DataLoader\n",
    "import torch\n",
    "from tqdm import tqdm\n",
    "\n",
    "class TeacherInferenceDataset(Dataset):\n",
    "    def __init__(self, dataframe, tokenizer, max_len=128):\n",
    "        self.texts = dataframe[\"text\"].tolist()\n",
    "        self.encodings = tokenizer(\n",
    "            self.texts,\n",
    "            truncation=True,\n",
    "            padding=True,\n",
    "            max_length=max_len\n",
    "        )\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.encodings[\"input_ids\"])\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "teacher_dataset = TeacherInferenceDataset(tesent_clean_df, tokenizer)\n",
    "\n",
    "teacher_loader = DataLoader(\n",
    "    teacher_dataset,\n",
    "    batch_size=32,\n",
    "    shuffle=False\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "all_teacher_logits = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for batch in tqdm(teacher_loader, desc=\"Teacher inference\"):\n",
    "        input_ids = batch[\"input_ids\"].to(device)\n",
    "        attention_mask = batch[\"attention_mask\"].to(device)\n",
    "\n",
    "        logits = teacher(input_ids, attention_mask)\n",
    "        all_teacher_logits.append(logits.cpu())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "teacher_logits = torch.cat(all_teacher_logits, dim=0)\n",
    "print(\"Teacher logits shape:\", teacher_logits.shape)  # (21119, 3)\n",
    "\n",
    "teacher_probs = torch.softmax(teacher_logits, dim=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "torch.save(teacher_probs, \"teacher_probs.pt\")\n",
    "print(\"✅ Saved teacher_probs.pt\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "assert len(tesent_clean_df) == teacher_probs.shape[0]\n",
    "print(\"✅ Teacher probs aligned with dataset\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch.nn.functional as F\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "teacher_probs = torch.load(\"teacher_probs.pt\").float()\n",
    "print(\"Loaded teacher_probs:\", teacher_probs.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "STUDENT_MODEL = \"distilbert-base-multilingual-cased\"\n",
    "\n",
    "student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)\n",
    "\n",
    "student = AutoModelForSequenceClassification.from_pretrained(\n",
    "    STUDENT_MODEL,\n",
    "    num_labels=3\n",
    ").to(device)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "class KDDataset(Dataset):\n",
    "    def __init__(self, dataframe, tokenizer, max_len=128):\n",
    "        self.texts = dataframe[\"text\"].tolist()\n",
    "        self.labels = dataframe[\"label\"].tolist()\n",
    "\n",
    "        self.encodings = tokenizer(\n",
    "            self.texts,\n",
    "            truncation=True,\n",
    "            padding=True,\n",
    "            max_length=max_len\n",
    "        )\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.labels)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}\n",
    "        item[\"label\"] = torch.tensor(self.labels[idx], dtype=torch.long)\n",
    "        item[\"idx\"] = idx\n",
    "        return item\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "train_dataset = KDDataset(tesent_clean_df, student_tokenizer)\n",
    "\n",
    "train_loader = DataLoader(\n",
    "    train_dataset,\n",
    "    batch_size=16,\n",
    "    shuffle=True\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "batch = next(iter(train_loader))\n",
    "print(batch.keys())\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def kd_loss(student_logits, teacher_probs, labels, alpha=0.7, T=2.0):\n",
    "    hard_loss = F.cross_entropy(student_logits, labels)\n",
    "\n",
    "    soft_loss = F.kl_div(\n",
    "        F.log_softmax(student_logits / T, dim=1),\n",
    "        F.softmax(teacher_probs / T, dim=1),\n",
    "        reduction=\"batchmean\"\n",
    "    ) * (T * T)\n",
    "\n",
    "    return alpha * hard_loss + (1 - alpha) * soft_loss\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "optimizer = torch.optim.AdamW(\n",
    "    student.parameters(),\n",
    "    lr=3e-5\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "epochs = 5\n",
    "student.train()\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    total_loss = 0\n",
    "\n",
    "    for batch in train_loader:\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        input_ids = batch[\"input_ids\"].to(device)\n",
    "        attention_mask = batch[\"attention_mask\"].to(device)\n",
    "        labels = batch[\"label\"].to(device)\n",
    "        idx = batch[\"idx\"]\n",
    "\n",
    "        outputs = student(\n",
    "            input_ids=input_ids,\n",
    "            attention_mask=attention_mask\n",
    "        )\n",
    "\n",
    "        student_logits = outputs.logits\n",
    "        batch_teacher_probs = teacher_probs[idx].to(device)\n",
    "\n",
    "        loss = kd_loss(\n",
    "            student_logits,\n",
    "            batch_teacher_probs,\n",
    "            labels\n",
    "        )\n",
    "\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        total_loss += loss.item()\n",
    "\n",
    "    print(f\"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / len(train_loader):.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "# Optional but recommended\n",
    "import torch\n",
    "torch.cuda.empty_cache()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "teacher_probs = torch.load(\"teacher_probs.pt\")\n",
    "print(teacher_probs.shape)  # MUST be (21119, 3)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# 0..21118 indices\n",
    "indices = list(range(len(tesent_clean_df)))\n",
    "\n",
    "train_idx, test_idx = train_test_split(\n",
    "    indices,\n",
    "    test_size=0.15,\n",
    "    random_state=42,\n",
    "    stratify=tesent_clean_df[\"label\"]\n",
    ")\n",
    "\n",
    "train_idx, val_idx = train_test_split(\n",
    "    train_idx,\n",
    "    test_size=0.1765,  # so val ≈ 15% of total\n",
    "    random_state=42,\n",
    "    stratify=tesent_clean_df.loc[train_idx, \"label\"]\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from torch.utils.data import Subset\n",
    "\n",
    "full_dataset = KDDataset(tesent_clean_df, student_tokenizer)\n",
    "\n",
    "train_dataset = Subset(full_dataset, train_idx)\n",
    "val_dataset   = Subset(full_dataset, val_idx)\n",
    "test_dataset  = Subset(full_dataset, test_idx)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from torch.utils.data import DataLoader\n",
    "\n",
    "train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)\n",
    "val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)\n",
    "test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
    "\n",
    "STUDENT_MODEL = \"distilbert-base-multilingual-cased\"\n",
    "\n",
    "student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)\n",
    "\n",
    "student = AutoModelForSequenceClassification.from_pretrained(\n",
    "    STUDENT_MODEL,\n",
    "    num_labels=3\n",
    ").to(device)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "optimizer = torch.optim.AdamW(student.parameters(), lr=3e-5)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch.nn.functional as F\n",
    "\n",
    "def evaluate(model, loader, device):\n",
    "    model.eval()\n",
    "    total_loss = 0.0\n",
    "    correct = 0\n",
    "    total = 0\n",
    "\n",
    "    with torch.no_grad():\n",
    "        for batch in loader:\n",
    "            input_ids = batch[\"input_ids\"].to(device)\n",
    "            attention_mask = batch[\"attention_mask\"].to(device)\n",
    "            labels = batch[\"label\"].to(device)\n",
    "\n",
    "            outputs = model(\n",
    "                input_ids=input_ids,\n",
    "                attention_mask=attention_mask\n",
    "            )\n",
    "\n",
    "            logits = outputs.logits  # ✅ important\n",
    "\n",
    "            loss = F.cross_entropy(logits, labels)\n",
    "            total_loss += loss.item()\n",
    "\n",
    "            preds = torch.argmax(logits, dim=1)\n",
    "            correct += (preds == labels).sum().item()\n",
    "            total += labels.size(0)\n",
    "\n",
    "    avg_loss = total_loss / len(loader)\n",
    "    acc = correct / total\n",
    "\n",
    "    return avg_loss, acc\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "best_val_acc = 0.0\n",
    "epochs = 5\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    student.train()\n",
    "    train_loss = 0.0\n",
    "\n",
    "    for batch in train_loader:\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        input_ids = batch[\"input_ids\"].to(device)\n",
    "        attention_mask = batch[\"attention_mask\"].to(device)\n",
    "        labels = batch[\"label\"].to(device)\n",
    "        idx = batch[\"idx\"]\n",
    "\n",
    "        outputs = student(\n",
    "            input_ids=input_ids,\n",
    "            attention_mask=attention_mask\n",
    "        )\n",
    "\n",
    "        student_logits = outputs.logits\n",
    "        batch_teacher_probs = teacher_probs[idx].to(device)\n",
    "\n",
    "        loss = kd_loss(\n",
    "            student_logits,\n",
    "            batch_teacher_probs,\n",
    "            labels\n",
    "        )\n",
    "\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        train_loss += loss.item()\n",
    "\n",
    "    train_loss /= len(train_loader)\n",
    "\n",
    "    # ✅ VALIDATION (correct call)\n",
    "    val_loss, val_acc = evaluate(student, val_loader, device)\n",
    "\n",
    "    print(\n",
    "        f\"Epoch {epoch+1}/{epochs} | \"\n",
    "        f\"Train Loss: {train_loss:.4f} | \"\n",
    "        f\"Val Loss: {val_loss:.4f} | \"\n",
    "        f\"Val Acc: {val_acc:.4f}\"\n",
    "    )\n",
    "\n",
    "    if val_acc > best_val_acc:\n",
    "        best_val_acc = val_acc\n",
    "        torch.save(student.state_dict(), \"best_student.pt\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "test_loss, test_acc = evaluate(student, test_loader, device)\n",
    "\n",
    "print(f\"TEST Loss: {test_loss:.4f}\")\n",
    "print(f\"TEST Accuracy: {test_acc:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "with open(\"test_results.txt\", \"w\") as f:\n",
    "    f.write(f\"Test Loss: {test_loss:.4f}\\n\")\n",
    "    f.write(f\"Test Accuracy: {test_acc:.4f}\\n\")\n",
    "\n",
    "print(\"Saved test_results.txt\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "torch.save(student.state_dict(), \"best_student.pt\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn.functional as F\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**INFERENCE TESTING**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn.functional as F\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "TEACHER_MODEL = \"google/muril-base-cased\"\n",
    "teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)\n",
    "\n",
    "teacher = MurilPEFTModel(base_model).to(device)\n",
    "teacher.load_state_dict(torch.load(\"best_muril_lora.pt\", map_location=device))\n",
    "teacher.eval()\n",
    "\n",
    "for p in teacher.parameters():\n",
    "    p.requires_grad = False\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "STUDENT_MODEL = \"distilbert-base-multilingual-cased\"\n",
    "student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)\n",
    "\n",
    "student = AutoModelForSequenceClassification.from_pretrained(\n",
    "    STUDENT_MODEL,\n",
    "    num_labels=3\n",
    ").to(device)\n",
    "\n",
    "student.load_state_dict(torch.load(\"best_student.pt\", map_location=device))\n",
    "student.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "id2label = {\n",
    "    0: \"Negative\",\n",
    "    1: \"Neutral\",\n",
    "    2: \"Positive\"\n",
    "}\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch.nn.functional as F\n",
    "\n",
    "def predict_with_confidence(model, tokenizer, text):\n",
    "    inputs = tokenizer(\n",
    "        text,\n",
    "        return_tensors=\"pt\",\n",
    "        truncation=True,\n",
    "        padding=True\n",
    "    ).to(device)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        outputs = model(\n",
    "            inputs[\"input_ids\"],\n",
    "            inputs[\"attention_mask\"]\n",
    "        )\n",
    "\n",
    "        # 🔑 KEY FIX: handle both cases\n",
    "        if hasattr(outputs, \"logits\"):\n",
    "            logits = outputs.logits          # student\n",
    "        else:\n",
    "            logits = outputs                 # teacher (already tensor)\n",
    "\n",
    "        probs = F.softmax(logits, dim=1)\n",
    "\n",
    "    conf, pred = torch.max(probs, dim=1)\n",
    "\n",
    "    return id2label[pred.item()], conf.item()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "sentences = {\n",
    "    \"Telugu\": [\n",
    "        \"ఈ సినిమా చాలా బాగుంది\",\n",
    "        \"ఈ సినిమా చాలా చెత్తగా ఉంది\",\n",
    "        \"ఈ సినిమా సరాసరిగా ఉంది\"\n",
    "    ],\n",
    "    \"Kannada\": [\n",
    "        \"ಈ ಸಿನಿಮಾ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ\",\n",
    "        \"ಈ ಸಿನಿಮಾ ತುಂಬಾ ಕೆಟ್ಟಾಗಿದೆ\",\n",
    "        \"ಈ ಸಿನಿಮಾ ಸರಾಸರಿ ಇದೆ\"\n",
    "    ],\n",
    "    \"Marathi\": [\n",
    "        \"हा चित्रपट खूप चांगला आहे\",\n",
    "        \"हा चित्रपट खूप वाईट आहे\",\n",
    "        \"हा चित्रपट ठीकठाक आहे\"\n",
    "    ]\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "for lang, sents in sentences.items():\n",
    "    print(f\"\\n===== {lang} =====\")\n",
    "    for sent in sents:\n",
    "        t_label, t_conf = predict_with_confidence(teacher, teacher_tokenizer, sent)\n",
    "        s_label, s_conf = predict_with_confidence(student, student_tokenizer, sent)\n",
    "\n",
    "        print(f\"\\nSentence: {sent}\")\n",
    "        print(f\"Teacher → {t_label} ({t_conf:.2f})\")\n",
    "        print(f\"Student → {s_label} ({s_conf:.2f})\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**OFFLINE KD FOR BiLstm**********"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn.functional as F\n",
    "\n",
    "teacher_probs = torch.load(\"teacher_probs.pt\").float()\n",
    "print(\"Teacher probs:\", teacher_probs.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(\"google/muril-base-cased\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from torch.utils.data import Dataset\n",
    "\n",
    "class KDDataset(Dataset):\n",
    "    def __init__(self, dataframe, tokenizer, max_len=128):\n",
    "        self.texts = dataframe[\"text\"].tolist()\n",
    "        self.labels = dataframe[\"label\"].tolist()\n",
    "\n",
    "        self.encodings = tokenizer(\n",
    "            self.texts,\n",
    "            truncation=True,\n",
    "            padding=True,\n",
    "            max_length=max_len\n",
    "        )\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.labels)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}\n",
    "        item[\"label\"] = torch.tensor(self.labels[idx], dtype=torch.long)\n",
    "        item[\"idx\"] = idx\n",
    "        return item\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch.nn as nn\n",
    "\n",
    "class BiLSTMStudent(nn.Module):\n",
    "    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256, num_classes=3):\n",
    "        super().__init__()\n",
    "\n",
    "        self.embedding = nn.Embedding(vocab_size, embed_dim)\n",
    "        self.lstm = nn.LSTM(\n",
    "            embed_dim,\n",
    "            hidden_dim,\n",
    "            batch_first=True,\n",
    "            bidirectional=True\n",
    "        )\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.fc = nn.Linear(hidden_dim * 2, num_classes)\n",
    "\n",
    "    def mean_pool(self, outputs, mask):\n",
    "        mask = mask.unsqueeze(-1)\n",
    "        summed = torch.sum(outputs * mask, dim=1)\n",
    "        counts = torch.clamp(mask.sum(dim=1), min=1e-9)\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "        x = self.embedding(input_ids)\n",
    "        outputs, _ = self.lstm(x)\n",
    "        pooled = self.mean_pool(outputs, attention_mask)\n",
    "        logits = self.fc(self.dropout(pooled))\n",
    "        return logits\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from torch.utils.data import DataLoader, Subset\n",
    "\n",
    "full_dataset = KDDataset(tesent_clean_df, tokenizer)\n",
    "\n",
    "train_dataset = Subset(full_dataset, train_idx)\n",
    "val_dataset   = Subset(full_dataset, val_idx)\n",
    "test_dataset  = Subset(full_dataset, test_idx)\n",
    "\n",
    "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
    "val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)\n",
    "test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def kd_loss(student_logits, teacher_probs, labels, alpha=0.7, T=2.0):\n",
    "    hard_loss = F.cross_entropy(student_logits, labels)\n",
    "\n",
    "    soft_loss = F.kl_div(\n",
    "        F.log_softmax(student_logits / T, dim=1),\n",
    "        F.softmax(teacher_probs / T, dim=1),\n",
    "        reduction=\"batchmean\"\n",
    "    ) * (T * T)\n",
    "\n",
    "    return alpha * hard_loss + (1 - alpha) * soft_loss\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def evaluate(model, loader, device):\n",
    "    model.eval()\n",
    "    total_loss = 0\n",
    "    correct = 0\n",
    "    total = 0\n",
    "\n",
    "    with torch.no_grad():\n",
    "        for batch in loader:\n",
    "            input_ids = batch[\"input_ids\"].to(device)\n",
    "            attention_mask = batch[\"attention_mask\"].to(device)\n",
    "            labels = batch[\"label\"].to(device)\n",
    "\n",
    "            logits = model(input_ids, attention_mask)\n",
    "\n",
    "            loss = F.cross_entropy(logits, labels)\n",
    "            total_loss += loss.item()\n",
    "\n",
    "            preds = torch.argmax(logits, dim=1)\n",
    "            correct += (preds == labels).sum().item()\n",
    "            total += labels.size(0)\n",
    "\n",
    "    return total_loss / len(loader), correct / total\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "student = BiLSTMStudent(\n",
    "    vocab_size=tokenizer.vocab_size\n",
    ").to(device)\n",
    "\n",
    "optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "best_val_acc = 0.0\n",
    "epochs = 8\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    student.train()\n",
    "    train_loss = 0\n",
    "\n",
    "    for batch in train_loader:\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        input_ids = batch[\"input_ids\"].to(device)\n",
    "        attention_mask = batch[\"attention_mask\"].to(device)\n",
    "        labels = batch[\"label\"].to(device)\n",
    "        idx = batch[\"idx\"]\n",
    "\n",
    "        student_logits = student(input_ids, attention_mask)\n",
    "        batch_teacher_probs = teacher_probs[idx].to(device)\n",
    "\n",
    "        loss = kd_loss(\n",
    "            student_logits,\n",
    "            batch_teacher_probs,\n",
    "            labels\n",
    "        )\n",
    "\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        train_loss += loss.item()\n",
    "\n",
    "    train_loss /= len(train_loader)\n",
    "    val_loss, val_acc = evaluate(student, val_loader, device)\n",
    "\n",
    "    print(\n",
    "        f\"Epoch {epoch+1} | \"\n",
    "        f\"Train Loss: {train_loss:.4f} | \"\n",
    "        f\"Val Acc: {val_acc:.4f}\"\n",
    "    )\n",
    "\n",
    "    if val_acc > best_val_acc:\n",
    "        best_val_acc = val_acc\n",
    "        torch.save(student.state_dict(), \"best_bilstm_offline_kd.pt\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "student.load_state_dict(\n",
    "    torch.load(\"best_bilstm_offline_kd.pt\")\n",
    ")\n",
    "\n",
    "test_loss, test_acc = evaluate(student, test_loader, device)\n",
    "\n",
    "print(f\"BiLSTM OFFLINE KD | Test Acc: {test_acc:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "tokenizer = AutoTokenizer.from_pretrained(\"google/muril-base-cased\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "student_bilstm = BiLSTMStudent(\n",
    "    vocab_size=tokenizer.vocab_size\n",
    ").to(device)\n",
    "\n",
    "student_bilstm.load_state_dict(\n",
    "    torch.load(\"best_bilstm_offline_kd.pt\", map_location=device)\n",
    ")\n",
    "\n",
    "student_bilstm.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "id2label = {\n",
    "    0: \"Negative\",\n",
    "    1: \"Neutral\",\n",
    "    2: \"Positive\"\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def predict_bilstm_with_confidence(model, tokenizer, text):\n",
    "    inputs = tokenizer(\n",
    "        text,\n",
    "        return_tensors=\"pt\",\n",
    "        truncation=True,\n",
    "        padding=True\n",
    "    ).to(device)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        logits = model(\n",
    "            inputs[\"input_ids\"],\n",
    "            inputs[\"attention_mask\"]\n",
    "        )\n",
    "        probs = F.softmax(logits, dim=1)\n",
    "\n",
    "    conf, pred = torch.max(probs, dim=1)\n",
    "\n",
    "    return id2label[pred.item()], conf.item()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def predict_teacher_with_confidence(model, tokenizer, text):\n",
    "    inputs = tokenizer(\n",
    "        text,\n",
    "        return_tensors=\"pt\",\n",
    "        truncation=True,\n",
    "        padding=True\n",
    "    ).to(device)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        logits = model(\n",
    "            inputs[\"input_ids\"],\n",
    "            inputs[\"attention_mask\"]\n",
    "        )\n",
    "        probs = F.softmax(logits, dim=1)\n",
    "\n",
    "    conf, pred = torch.max(probs, dim=1)\n",
    "\n",
    "    return id2label[pred.item()], conf.item()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "sentences = {\n",
    "    \"Telugu\": [\n",
    "        \"ఈ సినిమా చాలా బాగుంది\",\n",
    "        \"ఈ సినిమా చాలా చెత్తగా ఉంది\",\n",
    "        \"ఈ సినిమా సరాసరిగా ఉంది\"\n",
    "    ],\n",
    "    \"Kannada\": [\n",
    "        \"ಈ ಸಿನಿಮಾ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ\",\n",
    "        \"ಈ ಸಿನಿಮಾ ತುಂಬಾ ಕೆಟ್ಟಾಗಿದೆ\",\n",
    "        \"ಈ ಸಿನಿಮಾ ಸರಾಸರಿ ಇದೆ\"\n",
    "    ],\n",
    "    \"Marathi\": [\n",
    "        \"हा चित्रपट खूप चांगला आहे\",\n",
    "        \"हा चित्रपट खूप वाईट आहे\",\n",
    "        \"हा चित्रपट ठीकठाक आहे\"\n",
    "    ]\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "for lang, sents in sentences.items():\n",
    "    print(f\"\\n===== {lang} =====\")\n",
    "    for sent in sents:\n",
    "        t_label, t_conf = predict_teacher_with_confidence(\n",
    "            teacher, teacher_tokenizer, sent\n",
    "        )\n",
    "        b_label, b_conf = predict_bilstm_with_confidence(\n",
    "            student_bilstm, tokenizer, sent\n",
    "        )\n",
    "\n",
    "        print(f\"\\nSentence: {sent}\")\n",
    "        print(f\"Teacher → {t_label} ({t_conf:.2f})\")\n",
    "        print(f\"BiLSTM  → {b_label} ({b_conf:.2f})\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**BASELINE TESTING**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "from transformers import AutoModel, AutoTokenizer\n",
    "from peft import LoraConfig, get_peft_model, TaskType\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "model_name = \"google/muril-base-cased\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "\n",
    "# Base encoder\n",
    "base_model = AutoModel.from_pretrained(model_name)\n",
    "\n",
    "# LoRA config (MUST MATCH TRAINING)\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    target_modules=[\"query\",\"key\",\"value\",\"dense\"],\n",
    "    lora_dropout=0.1,\n",
    "    bias=\"none\",\n",
    "    task_type=TaskType.FEATURE_EXTRACTION\n",
    ")\n",
    "\n",
    "base_model = get_peft_model(base_model, lora_config)\n",
    "\n",
    "# Same model class\n",
    "class MurilPEFTModel(nn.Module):\n",
    "    def __init__(self, base_model, num_classes=3):\n",
    "        super().__init__()\n",
    "        self.base_model = base_model\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.classifier = nn.Linear(768, num_classes)\n",
    "        \n",
    "    def mean_pool(self, output, mask):\n",
    "        token_embeddings = output.last_hidden_state\n",
    "        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()\n",
    "        summed = torch.sum(token_embeddings * mask, 1)\n",
    "        counts = torch.clamp(mask.sum(1), min=1e-9)\n",
    "        return summed / counts\n",
    "    \n",
    "    def forward(self, input_ids, attention_mask):\n",
    "        output = self.base_model(\n",
    "            input_ids=input_ids,\n",
    "            attention_mask=attention_mask\n",
    "        )\n",
    "        pooled = self.mean_pool(output, attention_mask)\n",
    "        return self.classifier(self.dropout(pooled))\n",
    "\n",
    "model = MurilPEFTModel(base_model).to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "checkpoint_path = \"best_muril_lora.pt\"\n",
    "model.load_state_dict(torch.load(checkpoint_path, map_location=device))\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "model.base_model.print_trainable_parameters()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "next(model.parameters()).device"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, f1_score, classification_report\n",
    "\n",
    "all_preds, all_labels = [], []\n",
    "\n",
    "model.eval()\n",
    "with torch.no_grad():\n",
    "    for batch in test_loader:\n",
    "        input_ids = batch[\"input_ids\"].to(device)\n",
    "        attention_mask = batch[\"attention_mask\"].to(device)\n",
    "        labels = batch[\"label\"].to(device)\n",
    "\n",
    "        logits = model(input_ids, attention_mask)\n",
    "        preds = torch.argmax(logits, dim=1)\n",
    "\n",
    "        all_preds.extend(preds.cpu().numpy())\n",
    "        all_labels.extend(labels.cpu().numpy())\n",
    "\n",
    "accuracy = accuracy_score(all_labels, all_preds)\n",
    "macro_f1 = f1_score(all_labels, all_preds, average=\"macro\")\n",
    "weighted_f1 = f1_score(all_labels, all_preds, average=\"weighted\")\n",
    "\n",
    "accuracy, macro_f1, weighted_f1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**MEASURUNG EFFICIENCY METRiCS**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "pip install thop\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "from thop import profile\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
    "\n",
    "# Device (CPU is fine for FLOPs)\n",
    "device = torch.device(\"cpu\")\n",
    "\n",
    "# Base architecture (same as your student)\n",
    "model_name = \"distilbert-base-multilingual-cased\"\n",
    "\n",
    "# Recreate architecture\n",
    "model = AutoModelForSequenceClassification.from_pretrained(\n",
    "    model_name,\n",
    "    num_labels=3\n",
    ")\n",
    "\n",
    "# Load YOUR trained KD weights\n",
    "state_dict = torch.load(\"best_student.pt\", map_location=\"cpu\")\n",
    "model.load_state_dict(state_dict)\n",
    "\n",
    "model.to(device)\n",
    "model.eval()\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "\n",
    "# Dummy input (same seq length as evaluation)\n",
    "inputs = tokenizer(\n",
    "    \"ఈ సినిమా చాలా బాగుంది\",\n",
    "    return_tensors=\"pt\",\n",
    "    padding=\"max_length\",\n",
    "    truncation=True,\n",
    "    max_length=128\n",
    ")\n",
    "\n",
    "inputs = {k: v.to(device) for k, v in inputs.items()}\n",
    "\n",
    "# Measure FLOPs\n",
    "with torch.no_grad():\n",
    "    flops, params = profile(\n",
    "        model,\n",
    "        inputs=(inputs[\"input_ids\"], inputs[\"attention_mask\"]),\n",
    "        verbose=False\n",
    "    )\n",
    "\n",
    "print(f\"FLOPs per inference: {flops / 1e9:.2f} GFLOPs\")\n",
    "print(f\"Total parameters: {params / 1e6:.2f} M\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import time\n",
    "\n",
    "# Warm-up\n",
    "with torch.no_grad():\n",
    "    for _ in range(10):\n",
    "        _ = model(**inputs)\n",
    "\n",
    "# Measure latency\n",
    "runs = 100\n",
    "times = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for _ in range(runs):\n",
    "        start = time.time()\n",
    "        _ = model(**inputs)\n",
    "        end = time.time()\n",
    "        times.append((end - start) * 1000)  # ms\n",
    "\n",
    "avg_latency = sum(times) / len(times)\n",
    "\n",
    "print(f\"Average CPU inference latency (batch=1): {avg_latency:.2f} ms\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "class BiLSTMStudent(nn.Module):\n",
    "    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256, num_classes=3):\n",
    "        super().__init__()\n",
    "        self.embedding = nn.Embedding(vocab_size, embed_dim)\n",
    "        self.lstm = nn.LSTM(\n",
    "            embed_dim,\n",
    "            hidden_dim,\n",
    "            bidirectional=True,\n",
    "            batch_first=True\n",
    "        )\n",
    "        self.fc = nn.Linear(hidden_dim * 2, num_classes)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.embedding(x)\n",
    "        _, (h, _) = self.lstm(x)\n",
    "        h = torch.cat((h[-2], h[-1]), dim=1)\n",
    "        return self.fc(h)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "device = torch.device(\"cpu\")\n",
    "\n",
    "student = BiLSTMStudent(\n",
    "    vocab_size=197285,   # ✅ exact\n",
    "    embed_dim=300,\n",
    "    hidden_dim=256,\n",
    "    num_classes=3\n",
    ").to(device)\n",
    "\n",
    "state_dict = torch.load(\n",
    "    \"best_bilstm_offline_kd.pt\",\n",
    "    map_location=\"cpu\"\n",
    ")\n",
    "\n",
    "student.load_state_dict(state_dict)\n",
    "student.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from thop import profile\n",
    "import time\n",
    "\n",
    "SEQ_LEN = 128\n",
    "\n",
    "dummy_input = torch.randint(\n",
    "    0, 197285,\n",
    "    (1, SEQ_LEN),\n",
    "    device=device\n",
    ")\n",
    "\n",
    "with torch.no_grad():\n",
    "    flops, params = profile(student, inputs=(dummy_input,), verbose=False)\n",
    "\n",
    "print(f\"BiLSTM FLOPs per inference: {flops / 1e6:.2f} MFLOPs\")\n",
    "\n",
    "# CPU latency\n",
    "with torch.no_grad():\n",
    "    for _ in range(10):\n",
    "        _ = student(dummy_input)\n",
    "\n",
    "times = []\n",
    "for _ in range(100):\n",
    "    start = time.time()\n",
    "    _ = student(dummy_input)\n",
    "    times.append((time.time() - start) * 1000)\n",
    "\n",
    "print(f\"CPU latency (batch=1): {sum(times)/len(times):.2f} ms\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "total_params = sum(p.numel() for p in student.parameters())\n",
    "trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)\n",
    "\n",
    "print(f\"Total parameters: {total_params / 1e6:.2f} M\")\n",
    "print(f\"Trainable parameters: {trainable_params / 1e6:.2f} M\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from thop import profile\n",
    "\n",
    "model.eval()\n",
    "\n",
    "dummy_input_ids = torch.randint(\n",
    "    0, tokenizer.vocab_size,\n",
    "    (1, 128),\n",
    "    device=device\n",
    ")\n",
    "dummy_mask = torch.ones_like(dummy_input_ids)\n",
    "\n",
    "with torch.no_grad():\n",
    "    flops, params = profile(\n",
    "        model,\n",
    "        inputs=(dummy_input_ids, dummy_mask),\n",
    "        verbose=False\n",
    "    )\n",
    "\n",
    "print(f\"FLOPs per inference: {flops / 1e9:.2f} GFLOPs\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T12:13:39.756782Z",
     "iopub.status.busy": "2026-03-09T12:13:39.756472Z",
     "iopub.status.idle": "2026-03-09T12:13:43.203408Z",
     "shell.execute_reply": "2026-03-09T12:13:43.202637Z",
     "shell.execute_reply.started": "2026-03-09T12:13:39.756753Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "!pip install fvcore --quiet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-09T12:17:20.832675Z",
     "iopub.status.busy": "2026-03-09T12:17:20.832019Z",
     "iopub.status.idle": "2026-03-09T12:17:53.642368Z",
     "shell.execute_reply": "2026-03-09T12:17:53.641512Z",
     "shell.execute_reply.started": "2026-03-09T12:17:20.832642Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading teacher (MuRIL + LoRA)...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of BertForSequenceClassification were not initialized from the model checkpoint at google/muril-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Remapped 349/349 keys successfully.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ee08067d17b64b0f9f620426ffab5a44",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/206 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c1614bbaf2f445baaaf4037fe5de222c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "vocab.txt: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "71cdf67332f0460aa473d7dfa9b97e83",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "special_tokens_map.json:   0%|          | 0.00/113 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-multilingual-cased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✓ Teacher loaded\n",
      "\n",
      "Loading student (DistilBERT)...\n",
      "✓ Student loaded\n",
      "\n",
      "Measuring FLOPs...\n",
      "✓ FLOPs done\n",
      "\n",
      "Measuring latency (10 warmup + 100 runs)...\n",
      "✓ Latency done\n",
      "\n",
      "──────────────────────────────────────────────────────────────\n",
      "Metric                           MuRIL+LoRA       DistilBERT\n",
      "──────────────────────────────────────────────────────────────\n",
      "Total Params                       2065.31M          803.01M\n",
      "Trainable Params                      2.68M          135.33M\n",
      "FLOPs / Inference                   11.22G           5.44G\n",
      "CPU Latency mean (ms)               170.62           74.87\n",
      "CPU Latency std  (ms)                 5.23            3.59\n",
      "──────────────────────────────────────────────────────────────\n",
      "\n",
      "Compression Ratios:\n",
      "  Param reduction : 2.57x fewer parameters\n",
      "  FLOPs reduction : 2.06x fewer FLOPs\n",
      "  Speedup         : 2.28x faster on CPU\n"
     ]
    }
   ],
   "source": [
    "# ─── Cell 2: Full code with key remapping ────────────────────────────────────\n",
    "import torch\n",
    "import time\n",
    "import numpy as np\n",
    "from fvcore.nn import FlopCountAnalysis, parameter_count\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
    "from peft import LoraConfig, get_peft_model, TaskType\n",
    "\n",
    "device = torch.device(\"cpu\")\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "# HELPER FUNCTIONS\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "def make_inputs(tokenizer, device, seq_len=128):\n",
    "    enc = tokenizer(\n",
    "        \"ఈ సినిమా చాలా బాగుంది\",\n",
    "        return_tensors=\"pt\",\n",
    "        padding=\"max_length\",\n",
    "        truncation=True,\n",
    "        max_length=seq_len,\n",
    "    )\n",
    "    return enc[\"input_ids\"].to(device), enc[\"attention_mask\"].to(device)\n",
    "\n",
    "@torch.no_grad()\n",
    "def measure_flops(model, input_ids, attention_mask):\n",
    "    model.eval()\n",
    "    fc = FlopCountAnalysis(model, (input_ids, attention_mask))\n",
    "    fc.unsupported_ops_warnings(False)\n",
    "    fc.uncalled_modules_warnings(False)\n",
    "    return fc.total()\n",
    "\n",
    "@torch.no_grad()\n",
    "def measure_latency(model, input_ids, attention_mask, warmup=10, runs=100):\n",
    "    model.eval()\n",
    "    for _ in range(warmup):\n",
    "        model(input_ids, attention_mask=attention_mask)\n",
    "    times = []\n",
    "    for _ in range(runs):\n",
    "        t0 = time.perf_counter()\n",
    "        model(input_ids, attention_mask=attention_mask)\n",
    "        times.append((time.perf_counter() - t0) * 1000)\n",
    "    return np.mean(times), np.std(times)\n",
    "\n",
    "def remap_keys(state_dict, model):\n",
    "    \"\"\"\n",
    "    Automatically remap checkpoint keys to match the current model's keys.\n",
    "    Handles the double-wrapping issue where checkpoint has\n",
    "    'base_model.base_model.model.*' but model expects 'base_model.model.bert.*'\n",
    "    \"\"\"\n",
    "    model_keys  = set(model.state_dict().keys())\n",
    "    ckpt_keys   = set(state_dict.keys())\n",
    "\n",
    "    # If keys already match, return as-is\n",
    "    if ckpt_keys == model_keys:\n",
    "        print(\"  Keys match perfectly, no remapping needed.\")\n",
    "        return state_dict\n",
    "\n",
    "    new_state_dict = {}\n",
    "\n",
    "    for ckpt_key, value in state_dict.items():\n",
    "        # ── Rule 1: base_model.base_model.model.* → base_model.model.bert.* ──\n",
    "        if ckpt_key.startswith(\"base_model.base_model.model.\"):\n",
    "            suffix   = ckpt_key[len(\"base_model.base_model.model.\"):]\n",
    "            new_key  = f\"base_model.model.bert.{suffix}\"\n",
    "            if new_key in model_keys:\n",
    "                new_state_dict[new_key] = value\n",
    "                continue\n",
    "\n",
    "        # ── Rule 2: classifier.* → base_model.model.classifier.modules_to_save.default.* ──\n",
    "        if ckpt_key.startswith(\"classifier.\"):\n",
    "            suffix  = ckpt_key[len(\"classifier.\"):]\n",
    "            new_key = f\"base_model.model.classifier.modules_to_save.default.{suffix}\"\n",
    "            if new_key in model_keys:\n",
    "                new_state_dict[new_key] = value\n",
    "                # Also fill original_module copy\n",
    "                orig_key = f\"base_model.model.classifier.original_module.{suffix}\"\n",
    "                if orig_key in model_keys:\n",
    "                    new_state_dict[orig_key] = value\n",
    "                continue\n",
    "\n",
    "        # ── Fallback: keep original key if it exists in model ──\n",
    "        if ckpt_key in model_keys:\n",
    "            new_state_dict[ckpt_key] = value\n",
    "\n",
    "    # Report\n",
    "    matched   = len(new_state_dict)\n",
    "    total_mdl = len(model_keys)\n",
    "    print(f\"  Remapped {matched}/{total_mdl} keys successfully.\")\n",
    "\n",
    "    missing = model_keys - set(new_state_dict.keys())\n",
    "    if missing:\n",
    "        print(f\"  ⚠ Still missing {len(missing)} keys — loading with strict=False\")\n",
    "\n",
    "    return new_state_dict\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "# LOAD TEACHER — MuRIL + LoRA\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "print(\"Loading teacher (MuRIL + LoRA)...\")\n",
    "\n",
    "base_teacher = AutoModelForSequenceClassification.from_pretrained(\n",
    "    \"google/muril-base-cased\", num_labels=3\n",
    ")\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    lora_dropout=0.1,\n",
    "    target_modules=[\"query\", \"key\", \"value\", \"dense\"],\n",
    "    task_type=TaskType.SEQ_CLS,\n",
    "    bias=\"none\"\n",
    ")\n",
    "\n",
    "teacher = get_peft_model(base_teacher, lora_config)\n",
    "\n",
    "# Load and remap checkpoint\n",
    "raw_ckpt       = torch.load(\"best_muril_lora.pt\", map_location=\"cpu\")\n",
    "remapped_ckpt  = remap_keys(raw_ckpt, teacher)\n",
    "missing, unexpected = teacher.load_state_dict(remapped_ckpt, strict=False)\n",
    "\n",
    "if missing:\n",
    "    print(f\"  Missing  keys: {len(missing)}\")\n",
    "if unexpected:\n",
    "    print(f\"  Unexpected keys: {len(unexpected)}\")\n",
    "\n",
    "teacher.to(device).eval()\n",
    "muril_tok = AutoTokenizer.from_pretrained(\"google/muril-base-cased\")\n",
    "print(\"✓ Teacher loaded\\n\")\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "# LOAD STUDENT — DistilBERT\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "print(\"Loading student (DistilBERT)...\")\n",
    "\n",
    "student = AutoModelForSequenceClassification.from_pretrained(\n",
    "    \"distilbert-base-multilingual-cased\", num_labels=3\n",
    ")\n",
    "student.load_state_dict(\n",
    "    torch.load(\"best_student.pt\", map_location=\"cpu\"),\n",
    "    strict=True\n",
    ")\n",
    "student.to(device).eval()\n",
    "distil_tok = AutoTokenizer.from_pretrained(\"distilbert-base-multilingual-cased\")\n",
    "print(\"✓ Student loaded\\n\")\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "# PREPARE INPUTS\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "t_ids, t_mask = make_inputs(muril_tok,  device)\n",
    "s_ids, s_mask = make_inputs(distil_tok, device)\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "# MEASURE\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "print(\"Measuring FLOPs...\")\n",
    "teacher_flops = measure_flops(teacher, t_ids, t_mask)\n",
    "student_flops = measure_flops(student, s_ids, s_mask)\n",
    "print(\"✓ FLOPs done\\n\")\n",
    "\n",
    "print(\"Measuring latency (10 warmup + 100 runs)...\")\n",
    "t_lat, t_std = measure_latency(teacher, t_ids, t_mask)\n",
    "s_lat, s_std = measure_latency(student, s_ids, s_mask)\n",
    "print(\"✓ Latency done\\n\")\n",
    "\n",
    "teacher_params     = sum(parameter_count(teacher).values())\n",
    "teacher_trainable  = sum(p.numel() for p in teacher.parameters() if p.requires_grad)\n",
    "student_params     = sum(parameter_count(student).values())\n",
    "student_trainable  = sum(p.numel() for p in student.parameters() if p.requires_grad)\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "# RESULTS\n",
    "# ─────────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "sep = \"─\" * 62\n",
    "print(sep)\n",
    "print(f\"{'Metric':<26} {'MuRIL+LoRA':>16} {'DistilBERT':>16}\")\n",
    "print(sep)\n",
    "print(f\"{'Total Params':<26} {teacher_params/1e6:>15.2f}M {student_params/1e6:>15.2f}M\")\n",
    "print(f\"{'Trainable Params':<26} {teacher_trainable/1e6:>15.2f}M {student_trainable/1e6:>15.2f}M\")\n",
    "print(f\"{'FLOPs / Inference':<26} {teacher_flops/1e9:>14.2f}G {student_flops/1e9:>14.2f}G\")\n",
    "print(f\"{'CPU Latency mean (ms)':<26} {t_lat:>15.2f} {s_lat:>15.2f}\")\n",
    "print(f\"{'CPU Latency std  (ms)':<26} {t_std:>15.2f} {s_std:>15.2f}\")\n",
    "print(sep)\n",
    "print(f\"\\nCompression Ratios:\")\n",
    "print(f\"  Param reduction : {teacher_params/student_params:.2f}x fewer parameters\")\n",
    "print(f\"  FLOPs reduction : {teacher_flops/student_flops:.2f}x fewer FLOPs\")\n",
    "print(f\"  Speedup         : {t_lat/s_lat:.2f}x faster on CPU\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:00:37.351374Z",
     "iopub.status.busy": "2026-03-12T18:00:37.350959Z",
     "iopub.status.idle": "2026-03-12T18:00:39.229769Z",
     "shell.execute_reply": "2026-03-12T18:00:39.228963Z",
     "shell.execute_reply.started": "2026-03-12T18:00:37.351345Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'collections.OrderedDict'>\n",
      "odict_keys(['distilbert.embeddings.word_embeddings.weight', 'distilbert.embeddings.position_embeddings.weight', 'distilbert.embeddings.LayerNorm.weight', 'distilbert.embeddings.LayerNorm.bias', 'distilbert.transformer.layer.0.attention.q_lin.weight', 'distilbert.transformer.layer.0.attention.q_lin.bias', 'distilbert.transformer.layer.0.attention.k_lin.weight', 'distilbert.transformer.layer.0.attention.k_lin.bias', 'distilbert.transformer.layer.0.attention.v_lin.weight', 'distilbert.transformer.layer.0.attention.v_lin.bias', 'distilbert.transformer.layer.0.attention.out_lin.weight', 'distilbert.transformer.layer.0.attention.out_lin.bias', 'distilbert.transformer.layer.0.sa_layer_norm.weight', 'distilbert.transformer.layer.0.sa_layer_norm.bias', 'distilbert.transformer.layer.0.ffn.lin1.weight', 'distilbert.transformer.layer.0.ffn.lin1.bias', 'distilbert.transformer.layer.0.ffn.lin2.weight', 'distilbert.transformer.layer.0.ffn.lin2.bias', 'distilbert.transformer.layer.0.output_layer_norm.weight', 'distilbert.transformer.layer.0.output_layer_norm.bias', 'distilbert.transformer.layer.1.attention.q_lin.weight', 'distilbert.transformer.layer.1.attention.q_lin.bias', 'distilbert.transformer.layer.1.attention.k_lin.weight', 'distilbert.transformer.layer.1.attention.k_lin.bias', 'distilbert.transformer.layer.1.attention.v_lin.weight', 'distilbert.transformer.layer.1.attention.v_lin.bias', 'distilbert.transformer.layer.1.attention.out_lin.weight', 'distilbert.transformer.layer.1.attention.out_lin.bias', 'distilbert.transformer.layer.1.sa_layer_norm.weight', 'distilbert.transformer.layer.1.sa_layer_norm.bias', 'distilbert.transformer.layer.1.ffn.lin1.weight', 'distilbert.transformer.layer.1.ffn.lin1.bias', 'distilbert.transformer.layer.1.ffn.lin2.weight', 'distilbert.transformer.layer.1.ffn.lin2.bias', 'distilbert.transformer.layer.1.output_layer_norm.weight', 'distilbert.transformer.layer.1.output_layer_norm.bias', 'distilbert.transformer.layer.2.attention.q_lin.weight', 'distilbert.transformer.layer.2.attention.q_lin.bias', 'distilbert.transformer.layer.2.attention.k_lin.weight', 'distilbert.transformer.layer.2.attention.k_lin.bias', 'distilbert.transformer.layer.2.attention.v_lin.weight', 'distilbert.transformer.layer.2.attention.v_lin.bias', 'distilbert.transformer.layer.2.attention.out_lin.weight', 'distilbert.transformer.layer.2.attention.out_lin.bias', 'distilbert.transformer.layer.2.sa_layer_norm.weight', 'distilbert.transformer.layer.2.sa_layer_norm.bias', 'distilbert.transformer.layer.2.ffn.lin1.weight', 'distilbert.transformer.layer.2.ffn.lin1.bias', 'distilbert.transformer.layer.2.ffn.lin2.weight', 'distilbert.transformer.layer.2.ffn.lin2.bias', 'distilbert.transformer.layer.2.output_layer_norm.weight', 'distilbert.transformer.layer.2.output_layer_norm.bias', 'distilbert.transformer.layer.3.attention.q_lin.weight', 'distilbert.transformer.layer.3.attention.q_lin.bias', 'distilbert.transformer.layer.3.attention.k_lin.weight', 'distilbert.transformer.layer.3.attention.k_lin.bias', 'distilbert.transformer.layer.3.attention.v_lin.weight', 'distilbert.transformer.layer.3.attention.v_lin.bias', 'distilbert.transformer.layer.3.attention.out_lin.weight', 'distilbert.transformer.layer.3.attention.out_lin.bias', 'distilbert.transformer.layer.3.sa_layer_norm.weight', 'distilbert.transformer.layer.3.sa_layer_norm.bias', 'distilbert.transformer.layer.3.ffn.lin1.weight', 'distilbert.transformer.layer.3.ffn.lin1.bias', 'distilbert.transformer.layer.3.ffn.lin2.weight', 'distilbert.transformer.layer.3.ffn.lin2.bias', 'distilbert.transformer.layer.3.output_layer_norm.weight', 'distilbert.transformer.layer.3.output_layer_norm.bias', 'distilbert.transformer.layer.4.attention.q_lin.weight', 'distilbert.transformer.layer.4.attention.q_lin.bias', 'distilbert.transformer.layer.4.attention.k_lin.weight', 'distilbert.transformer.layer.4.attention.k_lin.bias', 'distilbert.transformer.layer.4.attention.v_lin.weight', 'distilbert.transformer.layer.4.attention.v_lin.bias', 'distilbert.transformer.layer.4.attention.out_lin.weight', 'distilbert.transformer.layer.4.attention.out_lin.bias', 'distilbert.transformer.layer.4.sa_layer_norm.weight', 'distilbert.transformer.layer.4.sa_layer_norm.bias', 'distilbert.transformer.layer.4.ffn.lin1.weight', 'distilbert.transformer.layer.4.ffn.lin1.bias', 'distilbert.transformer.layer.4.ffn.lin2.weight', 'distilbert.transformer.layer.4.ffn.lin2.bias', 'distilbert.transformer.layer.4.output_layer_norm.weight', 'distilbert.transformer.layer.4.output_layer_norm.bias', 'distilbert.transformer.layer.5.attention.q_lin.weight', 'distilbert.transformer.layer.5.attention.q_lin.bias', 'distilbert.transformer.layer.5.attention.k_lin.weight', 'distilbert.transformer.layer.5.attention.k_lin.bias', 'distilbert.transformer.layer.5.attention.v_lin.weight', 'distilbert.transformer.layer.5.attention.v_lin.bias', 'distilbert.transformer.layer.5.attention.out_lin.weight', 'distilbert.transformer.layer.5.attention.out_lin.bias', 'distilbert.transformer.layer.5.sa_layer_norm.weight', 'distilbert.transformer.layer.5.sa_layer_norm.bias', 'distilbert.transformer.layer.5.ffn.lin1.weight', 'distilbert.transformer.layer.5.ffn.lin1.bias', 'distilbert.transformer.layer.5.ffn.lin2.weight', 'distilbert.transformer.layer.5.ffn.lin2.bias', 'distilbert.transformer.layer.5.output_layer_norm.weight', 'distilbert.transformer.layer.5.output_layer_norm.bias', 'pre_classifier.weight', 'pre_classifier.bias', 'classifier.weight', 'classifier.bias'])\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "\n",
    "ckpt = torch.load(\"best_student.pt\", map_location=\"cpu\")\n",
    "print(type(ckpt))\n",
    "print(ckpt.keys() if isinstance(ckpt, dict) else \"not dict\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:01:20.997292Z",
     "iopub.status.busy": "2026-03-12T18:01:20.996779Z",
     "iopub.status.idle": "2026-03-12T18:01:30.066996Z",
     "shell.execute_reply": "2026-03-12T18:01:30.066183Z",
     "shell.execute_reply.started": "2026-03-12T18:01:20.997262Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m68.6/68.6 kB\u001b[0m \u001b[31m2.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m444.8/444.8 kB\u001b[0m \u001b[31m10.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.0/2.0 MB\u001b[0m \u001b[31m45.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m\n",
      "\u001b[?25h\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "google-adk 1.22.1 requires google-cloud-bigquery-storage>=2.0.0, which is not installed.\n",
      "langchain-core 0.3.79 requires packaging<26.0.0,>=23.2.0, but you have packaging 26.0rc2 which is incompatible.\n",
      "fastai 2.8.4 requires fastcore<1.9,>=1.8.0, but you have fastcore 1.11.3 which is incompatible.\u001b[0m\u001b[31m\n",
      "\u001b[0m"
     ]
    }
   ],
   "source": [
    "!pip install -q gradio transformers peft"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:10:25.806967Z",
     "iopub.status.busy": "2026-03-12T18:10:25.806623Z",
     "iopub.status.idle": "2026-03-12T18:10:32.988425Z",
     "shell.execute_reply": "2026-03-12T18:10:32.987743Z",
     "shell.execute_reply.started": "2026-03-12T18:10:25.806938Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading teacher...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-multilingual-cased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading DistilBERT student...\n",
      "Loading BiLSTM student...\n",
      "All models loaded successfully.\n",
      "* Running on local URL:  http://127.0.0.1:7860\n",
      "* Running on public URL: https://cd4d0fc1aa834dfc31.gradio.live\n",
      "\n",
      "This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"https://cd4d0fc1aa834dfc31.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import time\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import gradio as gr\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from transformers import (\n",
    "    AutoTokenizer,\n",
    "    AutoModel,\n",
    "    DistilBertForSequenceClassification\n",
    ")\n",
    "from peft import LoraConfig, get_peft_model\n",
    "\n",
    "# =========================\n",
    "# CONFIG\n",
    "# =========================\n",
    "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "MAX_LEN = 128\n",
    "NUM_CLASSES = 3\n",
    "TEACHER_MODEL = \"google/muril-base-cased\"\n",
    "DISTIL_MODEL_NAME = \"distilbert-base-multilingual-cased\"\n",
    "\n",
    "LABEL_MAP = {\n",
    "    0: \"Negative\",\n",
    "    1: \"Neutral\",\n",
    "    2: \"Positive\"\n",
    "}\n",
    "\n",
    "# =========================\n",
    "# TOKENIZER\n",
    "# =========================\n",
    "tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)\n",
    "\n",
    "# =========================\n",
    "# MODEL DEFINITIONS\n",
    "# =========================\n",
    "class TeacherClassifier(nn.Module):\n",
    "    def __init__(self, base_model_name, num_classes):\n",
    "        super().__init__()\n",
    "        base_teacher = AutoModel.from_pretrained(base_model_name)\n",
    "\n",
    "        lora_config = LoraConfig(\n",
    "            r=8,\n",
    "            lora_alpha=16,\n",
    "            target_modules=[\"query\", \"key\", \"value\"],\n",
    "            lora_dropout=0.1,\n",
    "            bias=\"none\"\n",
    "        )\n",
    "\n",
    "        self.encoder = get_peft_model(base_teacher, lora_config)\n",
    "        self.classifier = nn.Linear(768, num_classes)\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)\n",
    "        cls = outputs.last_hidden_state[:, 0, :]\n",
    "        logits = self.classifier(cls)\n",
    "        return logits\n",
    "\n",
    "\n",
    "class BiLSTMStudent(nn.Module):\n",
    "    def __init__(self, vocab_size, num_classes):\n",
    "        super().__init__()\n",
    "\n",
    "        self.embedding = nn.Embedding(vocab_size, 300)\n",
    "\n",
    "        self.lstm = nn.LSTM(\n",
    "            input_size=300,\n",
    "            hidden_size=256,\n",
    "            num_layers=1,\n",
    "            batch_first=True,\n",
    "            bidirectional=True\n",
    "        )\n",
    "\n",
    "        self.fc = nn.Linear(512, num_classes)\n",
    "\n",
    "    def forward(self, input_ids, attention_mask=None):\n",
    "        x = self.embedding(input_ids)\n",
    "        out, _ = self.lstm(x)\n",
    "        out = out[:, -1, :]\n",
    "        logits = self.fc(out)\n",
    "        return logits\n",
    "\n",
    "\n",
    "# =========================\n",
    "# LOAD MODELS\n",
    "# =========================\n",
    "print(\"Loading teacher...\")\n",
    "teacher_model = TeacherClassifier(TEACHER_MODEL, NUM_CLASSES)\n",
    "\n",
    "teacher_state = torch.load(\"best_muril_lora.pt\", map_location=DEVICE)\n",
    "teacher_model.load_state_dict(teacher_state, strict=False)\n",
    "teacher_model.to(DEVICE)\n",
    "teacher_model.eval()\n",
    "\n",
    "print(\"Loading DistilBERT student...\")\n",
    "distil_model = DistilBertForSequenceClassification.from_pretrained(\n",
    "    DISTIL_MODEL_NAME,\n",
    "    num_labels=NUM_CLASSES\n",
    ")\n",
    "\n",
    "distil_state = torch.load(\"best_student.pt\", map_location=DEVICE)\n",
    "distil_model.load_state_dict(distil_state)\n",
    "distil_model.to(DEVICE)\n",
    "distil_model.eval()\n",
    "\n",
    "print(\"Loading BiLSTM student...\")\n",
    "bilstm_model = BiLSTMStudent(tokenizer.vocab_size, NUM_CLASSES)\n",
    "\n",
    "bilstm_state = torch.load(\"best_bilstm_offline_kd.pt\", map_location=DEVICE)\n",
    "bilstm_model.load_state_dict(bilstm_state)\n",
    "bilstm_model.to(DEVICE)\n",
    "bilstm_model.eval()\n",
    "\n",
    "print(\"All models loaded successfully.\")\n",
    "\n",
    "# =========================\n",
    "# HELPERS\n",
    "# =========================\n",
    "def encode_text(text):\n",
    "    enc = tokenizer(\n",
    "        text,\n",
    "        padding=\"max_length\",\n",
    "        truncation=True,\n",
    "        max_length=MAX_LEN,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "    return {\n",
    "        \"input_ids\": enc[\"input_ids\"].to(DEVICE),\n",
    "        \"attention_mask\": enc[\"attention_mask\"].to(DEVICE)\n",
    "    }\n",
    "\n",
    "\n",
    "def probs_to_dict(probs):\n",
    "    return {\n",
    "        \"Negative\": float(probs[0]),\n",
    "        \"Neutral\": float(probs[1]),\n",
    "        \"Positive\": float(probs[2]),\n",
    "    }\n",
    "\n",
    "\n",
    "def format_single_result(model_name, probs, elapsed_ms):\n",
    "    pred_idx = int(probs.argmax())\n",
    "    pred_label = LABEL_MAP[pred_idx]\n",
    "    confidence = float(probs[pred_idx])\n",
    "\n",
    "    result = (\n",
    "        f\"**Model:** {model_name}\\n\\n\"\n",
    "        f\"**Prediction:** {pred_label}\\n\\n\"\n",
    "        f\"**Confidence:** {confidence:.4f}\\n\\n\"\n",
    "        f\"**Latency:** {elapsed_ms:.2f} ms\"\n",
    "    )\n",
    "    return result\n",
    "\n",
    "\n",
    "def make_prob_chart(prob_dict, title):\n",
    "    fig, ax = plt.subplots(figsize=(6, 4))\n",
    "    ax.bar(list(prob_dict.keys()), list(prob_dict.values()))\n",
    "    ax.set_ylim(0, 1)\n",
    "    ax.set_ylabel(\"Probability\")\n",
    "    ax.set_title(title)\n",
    "    plt.tight_layout()\n",
    "    return fig\n",
    "\n",
    "\n",
    "def make_comparison_chart(teacher_probs, distil_probs, bilstm_probs):\n",
    "    labels = [\"Negative\", \"Neutral\", \"Positive\"]\n",
    "    x = list(range(len(labels)))\n",
    "    width = 0.25\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(8, 4))\n",
    "    ax.bar([i - width for i in x], teacher_probs, width=width, label=\"Teacher\")\n",
    "    ax.bar(x, distil_probs, width=width, label=\"DistilBERT\")\n",
    "    ax.bar([i + width for i in x], bilstm_probs, width=width, label=\"BiLSTM\")\n",
    "\n",
    "    ax.set_xticks(x)\n",
    "    ax.set_xticklabels(labels)\n",
    "    ax.set_ylim(0, 1)\n",
    "    ax.set_ylabel(\"Probability\")\n",
    "    ax.set_title(\"Model Comparison\")\n",
    "    ax.legend()\n",
    "    plt.tight_layout()\n",
    "    return fig\n",
    "\n",
    "\n",
    "# =========================\n",
    "# PREDICTION FUNCTIONS\n",
    "# =========================\n",
    "@torch.no_grad()\n",
    "def predict_teacher(text):\n",
    "    inputs = encode_text(text)\n",
    "    start = time.time()\n",
    "    logits = teacher_model(inputs[\"input_ids\"], inputs[\"attention_mask\"])\n",
    "    elapsed_ms = (time.time() - start) * 1000\n",
    "    probs = F.softmax(logits, dim=1).cpu().numpy()[0]\n",
    "    return probs, elapsed_ms\n",
    "\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_distilbert(text):\n",
    "    inputs = encode_text(text)\n",
    "    start = time.time()\n",
    "    logits = distil_model(**inputs).logits\n",
    "    elapsed_ms = (time.time() - start) * 1000\n",
    "    probs = F.softmax(logits, dim=1).cpu().numpy()[0]\n",
    "    return probs, elapsed_ms\n",
    "\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_bilstm(text):\n",
    "    inputs = encode_text(text)\n",
    "    start = time.time()\n",
    "    logits = bilstm_model(inputs[\"input_ids\"], inputs[\"attention_mask\"])\n",
    "    elapsed_ms = (time.time() - start) * 1000\n",
    "    probs = F.softmax(logits, dim=1).cpu().numpy()[0]\n",
    "    return probs, elapsed_ms\n",
    "\n",
    "\n",
    "def run_selected_model(text, model_name, language):\n",
    "    if not text or not text.strip():\n",
    "        return \"Please enter some text.\", None\n",
    "\n",
    "    if model_name == \"Teacher (MuRIL + LoRA)\":\n",
    "        probs, elapsed = predict_teacher(text)\n",
    "        result_text = format_single_result(model_name, probs, elapsed)\n",
    "        fig = make_prob_chart(probs_to_dict(probs), f\"{model_name} Probabilities\")\n",
    "        return result_text, fig\n",
    "\n",
    "    elif model_name == \"Student (DistilBERT)\":\n",
    "        probs, elapsed = predict_distilbert(text)\n",
    "        result_text = format_single_result(model_name, probs, elapsed)\n",
    "        fig = make_prob_chart(probs_to_dict(probs), f\"{model_name} Probabilities\")\n",
    "        return result_text, fig\n",
    "\n",
    "    elif model_name == \"Student (BiLSTM)\":\n",
    "        probs, elapsed = predict_bilstm(text)\n",
    "        result_text = format_single_result(model_name, probs, elapsed)\n",
    "        fig = make_prob_chart(probs_to_dict(probs), f\"{model_name} Probabilities\")\n",
    "        return result_text, fig\n",
    "\n",
    "    else:\n",
    "        return \"Invalid model selected.\", None\n",
    "\n",
    "\n",
    "def compare_all_models(text, language):\n",
    "    if not text or not text.strip():\n",
    "        return None, None\n",
    "\n",
    "    teacher_probs, teacher_time = predict_teacher(text)\n",
    "    distil_probs, distil_time = predict_distilbert(text)\n",
    "    bilstm_probs, bilstm_time = predict_bilstm(text)\n",
    "\n",
    "    rows = []\n",
    "    for model_name, probs, t in [\n",
    "        (\"Teacher (MuRIL + LoRA)\", teacher_probs, teacher_time),\n",
    "        (\"Student (DistilBERT)\", distil_probs, distil_time),\n",
    "        (\"Student (BiLSTM)\", bilstm_probs, bilstm_time),\n",
    "    ]:\n",
    "        pred_idx = int(probs.argmax())\n",
    "        rows.append({\n",
    "            \"Model\": model_name,\n",
    "            \"Prediction\": LABEL_MAP[pred_idx],\n",
    "            \"Confidence\": round(float(probs[pred_idx]), 4),\n",
    "            \"Negative\": round(float(probs[0]), 4),\n",
    "            \"Neutral\": round(float(probs[1]), 4),\n",
    "            \"Positive\": round(float(probs[2]), 4),\n",
    "            \"Latency (ms)\": round(float(t), 2),\n",
    "        })\n",
    "\n",
    "    df = pd.DataFrame(rows)\n",
    "    fig = make_comparison_chart(teacher_probs, distil_probs, bilstm_probs)\n",
    "    return df, fig\n",
    "\n",
    "\n",
    "# =========================\n",
    "# EXAMPLES\n",
    "# =========================\n",
    "examples = [\n",
    "    [\"ఈ సినిమా చాలా బాగుంది\", \"Teacher (MuRIL + LoRA)\", \"Telugu\"],\n",
    "    [\"ಈ ಸಿನಿಮಾ ಚೆನ್ನಾಗಿದೆ\", \"Student (DistilBERT)\", \"Kannada\"],\n",
    "    [\"हा चित्रपट खूप छान आहे\", \"Student (BiLSTM)\", \"Marathi\"],\n",
    "]\n",
    "\n",
    "# =========================\n",
    "# GRADIO UI\n",
    "# =========================\n",
    "with gr.Blocks(title=\"Indic Multilingual Sentiment Analysis Demo\") as demo:\n",
    "    gr.Markdown(\"# Low-Cost System for Multilingual Sentiment Analysis in Indic Languages\")\n",
    "    gr.Markdown(\n",
    "        \"Train on **Telugu**, test/demo on **Telugu, Kannada, Marathi**. \"\n",
    "        \"Models: **MuRIL + LoRA Teacher**, **DistilBERT Student**, **BiLSTM Student**.\"\n",
    "    )\n",
    "\n",
    "    with gr.Row():\n",
    "        with gr.Column():\n",
    "            text_input = gr.Textbox(\n",
    "                label=\"Enter text\",\n",
    "                lines=4,\n",
    "                placeholder=\"Type Telugu / Kannada / Marathi text here...\"\n",
    "            )\n",
    "\n",
    "            model_choice = gr.Dropdown(\n",
    "                choices=[\n",
    "                    \"Teacher (MuRIL + LoRA)\",\n",
    "                    \"Student (DistilBERT)\",\n",
    "                    \"Student (BiLSTM)\"\n",
    "                ],\n",
    "                value=\"Teacher (MuRIL + LoRA)\",\n",
    "                label=\"Choose model\"\n",
    "            )\n",
    "\n",
    "            language_choice = gr.Dropdown(\n",
    "                choices=[\"Telugu\", \"Kannada\", \"Marathi\"],\n",
    "                value=\"Telugu\",\n",
    "                label=\"Language\"\n",
    "            )\n",
    "\n",
    "            with gr.Row():\n",
    "                predict_btn = gr.Button(\"Predict\")\n",
    "                compare_btn = gr.Button(\"Compare All Models\")\n",
    "\n",
    "        with gr.Column():\n",
    "            result_output = gr.Markdown(label=\"Prediction\")\n",
    "            chart_output = gr.Plot(label=\"Probability Chart\")\n",
    "\n",
    "    gr.Markdown(\"## Model Comparison\")\n",
    "    comparison_table = gr.Dataframe(\n",
    "        headers=[\"Model\", \"Prediction\", \"Confidence\", \"Negative\", \"Neutral\", \"Positive\", \"Latency (ms)\"],\n",
    "        label=\"Comparison Table\"\n",
    "    )\n",
    "    comparison_chart = gr.Plot(label=\"Comparison Chart\")\n",
    "\n",
    "    gr.Examples(\n",
    "        examples=examples,\n",
    "        inputs=[text_input, model_choice, language_choice]\n",
    "    )\n",
    "\n",
    "    predict_btn.click(\n",
    "        fn=run_selected_model,\n",
    "        inputs=[text_input, model_choice, language_choice],\n",
    "        outputs=[result_output, chart_output]\n",
    "    )\n",
    "\n",
    "    compare_btn.click(\n",
    "        fn=compare_all_models,\n",
    "        inputs=[text_input, language_choice],\n",
    "        outputs=[comparison_table, comparison_chart]\n",
    "    )\n",
    "\n",
    "demo.launch()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:29:46.352941Z",
     "iopub.status.busy": "2026-03-12T18:29:46.352611Z",
     "iopub.status.idle": "2026-03-12T18:29:46.359260Z",
     "shell.execute_reply": "2026-03-12T18:29:46.358656Z",
     "shell.execute_reply.started": "2026-03-12T18:29:46.352913Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "class MurilPEFTModel(nn.Module):\n",
    "\n",
    "    def __init__(self, base_model, num_classes=3):\n",
    "\n",
    "        super().__init__()\n",
    "\n",
    "        self.base_model = base_model\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.classifier = nn.Linear(768, num_classes)\n",
    "\n",
    "    def mean_pool(self, output, mask):\n",
    "\n",
    "        token_embeddings = output.last_hidden_state\n",
    "\n",
    "        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()\n",
    "\n",
    "        summed = torch.sum(token_embeddings * mask, 1)\n",
    "\n",
    "        counts = torch.clamp(mask.sum(1), min=1e-9)\n",
    "\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "\n",
    "        output = self.base_model(\n",
    "            input_ids=input_ids,\n",
    "            attention_mask=attention_mask\n",
    "        )\n",
    "\n",
    "        pooled = self.mean_pool(output, attention_mask)\n",
    "\n",
    "        logits = self.classifier(self.dropout(pooled))\n",
    "\n",
    "        return logits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:29:59.484014Z",
     "iopub.status.busy": "2026-03-12T18:29:59.483407Z",
     "iopub.status.idle": "2026-03-12T18:30:02.045540Z",
     "shell.execute_reply": "2026-03-12T18:30:02.044769Z",
     "shell.execute_reply.started": "2026-03-12T18:29:59.483984Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading teacher...\n",
      "Teacher loaded correctly\n"
     ]
    }
   ],
   "source": [
    "from transformers import AutoModel\n",
    "from peft import LoraConfig, get_peft_model, TaskType\n",
    "\n",
    "print(\"Loading teacher...\")\n",
    "\n",
    "base_model = AutoModel.from_pretrained(\"google/muril-base-cased\")\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    target_modules=[\"query\",\"key\",\"value\",\"dense\"],\n",
    "    lora_dropout=0.1,\n",
    "    bias=\"none\",\n",
    "    task_type=TaskType.FEATURE_EXTRACTION\n",
    ")\n",
    "\n",
    "base_model = get_peft_model(base_model, lora_config)\n",
    "\n",
    "teacher_model = MurilPEFTModel(base_model).to(DEVICE)\n",
    "\n",
    "state = torch.load(\"best_muril_lora.pt\", map_location=DEVICE)\n",
    "\n",
    "teacher_model.load_state_dict(state)\n",
    "\n",
    "teacher_model.eval()\n",
    "\n",
    "print(\"Teacher loaded correctly\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:31:31.843970Z",
     "iopub.status.busy": "2026-03-12T18:31:31.843248Z",
     "iopub.status.idle": "2026-03-12T18:31:31.849613Z",
     "shell.execute_reply": "2026-03-12T18:31:31.848848Z",
     "shell.execute_reply.started": "2026-03-12T18:31:31.843940Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def predict_teacher(text):\n",
    "\n",
    "    enc = tokenizer(\n",
    "        text,\n",
    "        padding=\"max_length\",\n",
    "        truncation=True,\n",
    "        max_length=128,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "\n",
    "    # remove token_type_ids\n",
    "    enc = {k:v.to(DEVICE) for k,v in enc.items() if k != \"token_type_ids\"}\n",
    "\n",
    "    with torch.no_grad():\n",
    "\n",
    "        logits = teacher_model(**enc)\n",
    "\n",
    "        probs = torch.softmax(logits,dim=1)\n",
    "\n",
    "    probs = probs.cpu().numpy()[0]\n",
    "\n",
    "    pred = probs.argmax()\n",
    "\n",
    "    labels = [\"Negative\",\"Neutral\",\"Positive\"]\n",
    "\n",
    "    print(\"Input:\", text)\n",
    "    print(\"Prediction:\", labels[pred])\n",
    "\n",
    "    for i,p in enumerate(probs):\n",
    "        print(labels[i],\":\",round(float(p),4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:31:35.062624Z",
     "iopub.status.busy": "2026-03-12T18:31:35.062012Z",
     "iopub.status.idle": "2026-03-12T18:31:35.202207Z",
     "shell.execute_reply": "2026-03-12T18:31:35.201581Z",
     "shell.execute_reply.started": "2026-03-12T18:31:35.062592Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input: ఈ సినిమా చాలా బాగుంది\n",
      "Prediction: Positive\n",
      "Negative : 0.0594\n",
      "Neutral : 0.0798\n",
      "Positive : 0.8607\n",
      "Input: ఈ సినిమా చాలా చెడ్డది\n",
      "Prediction: Negative\n",
      "Negative : 0.8439\n",
      "Neutral : 0.0916\n",
      "Positive : 0.0645\n"
     ]
    }
   ],
   "source": [
    "predict_teacher(\"ఈ సినిమా చాలా బాగుంది\")\n",
    "predict_teacher(\"ఈ సినిమా చాలా చెడ్డది\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:46:11.941734Z",
     "iopub.status.busy": "2026-03-12T18:46:11.941434Z",
     "iopub.status.idle": "2026-03-12T18:46:13.132213Z",
     "shell.execute_reply": "2026-03-12T18:46:13.131473Z",
     "shell.execute_reply.started": "2026-03-12T18:46:11.941707Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-multilingual-cased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading DistilBERT student...\n",
      "DistilBERT loaded correctly\n"
     ]
    }
   ],
   "source": [
    "from transformers import AutoModelForSequenceClassification\n",
    "\n",
    "print(\"Loading DistilBERT student...\")\n",
    "\n",
    "distil_model = AutoModelForSequenceClassification.from_pretrained(\n",
    "    \"distilbert-base-multilingual-cased\",\n",
    "    num_labels=3\n",
    ")\n",
    "\n",
    "state = torch.load(\n",
    "    \"best_student.pt\",\n",
    "    map_location=DEVICE\n",
    ")\n",
    "\n",
    "distil_model.load_state_dict(state)\n",
    "\n",
    "distil_model.to(DEVICE)\n",
    "distil_model.eval()\n",
    "\n",
    "print(\"DistilBERT loaded correctly\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:46:16.166766Z",
     "iopub.status.busy": "2026-03-12T18:46:16.166466Z",
     "iopub.status.idle": "2026-03-12T18:46:16.629594Z",
     "shell.execute_reply": "2026-03-12T18:46:16.628963Z",
     "shell.execute_reply.started": "2026-03-12T18:46:16.166740Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer\n",
    "\n",
    "STUDENT_MODEL = \"distilbert-base-multilingual-cased\"\n",
    "\n",
    "student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:46:19.811904Z",
     "iopub.status.busy": "2026-03-12T18:46:19.811050Z",
     "iopub.status.idle": "2026-03-12T18:46:19.817341Z",
     "shell.execute_reply": "2026-03-12T18:46:19.816725Z",
     "shell.execute_reply.started": "2026-03-12T18:46:19.811870Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def predict_distilbert(text):\n",
    "\n",
    "    enc = student_tokenizer(\n",
    "        text,\n",
    "        truncation=True,\n",
    "        padding=True,\n",
    "        max_length=128,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "\n",
    "    enc = {k:v.to(DEVICE) for k,v in enc.items() if k != \"token_type_ids\"}\n",
    "\n",
    "    with torch.no_grad():\n",
    "        outputs = distil_model(**enc)\n",
    "        logits = outputs.logits\n",
    "        probs = torch.softmax(logits, dim=1)\n",
    "\n",
    "    probs = probs.cpu().numpy()[0]\n",
    "\n",
    "    labels = [\"Negative\",\"Neutral\",\"Positive\"]\n",
    "\n",
    "    pred = probs.argmax()\n",
    "\n",
    "    print(\"Input:\", text)\n",
    "    print(\"Prediction:\", labels[pred])\n",
    "\n",
    "    for i,p in enumerate(probs):\n",
    "        print(labels[i],\":\",round(float(p),4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:46:24.226610Z",
     "iopub.status.busy": "2026-03-12T18:46:24.226309Z",
     "iopub.status.idle": "2026-03-12T18:46:24.242855Z",
     "shell.execute_reply": "2026-03-12T18:46:24.242086Z",
     "shell.execute_reply.started": "2026-03-12T18:46:24.226582Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input: ఈ సినిమా చాలా బాగుంది\n",
      "Prediction: Positive\n",
      "Negative : 0.0705\n",
      "Neutral : 0.0894\n",
      "Positive : 0.8401\n",
      "Input: ఈ సినిమా చాలా చెడ్డది\n",
      "Prediction: Positive\n",
      "Negative : 0.0766\n",
      "Neutral : 0.2325\n",
      "Positive : 0.6909\n"
     ]
    }
   ],
   "source": [
    "predict_distilbert(\"ఈ సినిమా చాలా బాగుంది\")\n",
    "predict_distilbert(\"ఈ సినిమా చాలా చెడ్డది\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:47:17.367287Z",
     "iopub.status.busy": "2026-03-12T18:47:17.366975Z",
     "iopub.status.idle": "2026-03-12T18:47:17.395575Z",
     "shell.execute_reply": "2026-03-12T18:47:17.394967Z",
     "shell.execute_reply.started": "2026-03-12T18:47:17.367260Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input: ఈ సినిమా చాలా చెడ్డది\n",
      "Prediction: Negative\n",
      "Negative : 0.8439\n",
      "Neutral : 0.0916\n",
      "Positive : 0.0645\n"
     ]
    }
   ],
   "source": [
    "predict_teacher(\"ఈ సినిమా చాలా చెడ్డది\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:53:47.561182Z",
     "iopub.status.busy": "2026-03-12T18:53:47.560817Z",
     "iopub.status.idle": "2026-03-12T18:53:49.016565Z",
     "shell.execute_reply": "2026-03-12T18:53:49.015767Z",
     "shell.execute_reply.started": "2026-03-12T18:53:47.561151Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading BiLSTM student...\n",
      "BiLSTM loaded correctly\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "\n",
    "print(\"Loading BiLSTM student...\")\n",
    "\n",
    "class BiLSTMStudent(nn.Module):\n",
    "    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256, num_classes=3):\n",
    "        super().__init__()\n",
    "\n",
    "        self.embedding = nn.Embedding(vocab_size, embed_dim)\n",
    "\n",
    "        self.lstm = nn.LSTM(\n",
    "            embed_dim,\n",
    "            hidden_dim,\n",
    "            batch_first=True,\n",
    "            bidirectional=True\n",
    "        )\n",
    "\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "\n",
    "        self.fc = nn.Linear(hidden_dim * 2, num_classes)\n",
    "\n",
    "    def mean_pool(self, outputs, mask):\n",
    "        mask = mask.unsqueeze(-1)\n",
    "        summed = torch.sum(outputs * mask, dim=1)\n",
    "        counts = torch.clamp(mask.sum(dim=1), min=1e-9)\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "\n",
    "        x = self.embedding(input_ids)\n",
    "\n",
    "        outputs, _ = self.lstm(x)\n",
    "\n",
    "        pooled = self.mean_pool(outputs, attention_mask)\n",
    "\n",
    "        logits = self.fc(self.dropout(pooled))\n",
    "\n",
    "        return logits\n",
    "\n",
    "\n",
    "# recreate tokenizer used during training\n",
    "bilstm_tokenizer = AutoTokenizer.from_pretrained(\"google/muril-base-cased\")\n",
    "\n",
    "# build model\n",
    "bilstm_model = BiLSTMStudent(\n",
    "    vocab_size=bilstm_tokenizer.vocab_size\n",
    ").to(DEVICE)\n",
    "\n",
    "# load checkpoint\n",
    "state = torch.load(\n",
    "    \"best_bilstm_offline_kd.pt\",\n",
    "    map_location=DEVICE\n",
    ")\n",
    "\n",
    "bilstm_model.load_state_dict(state)\n",
    "\n",
    "bilstm_model.eval()\n",
    "\n",
    "print(\"BiLSTM loaded correctly\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:53:58.826456Z",
     "iopub.status.busy": "2026-03-12T18:53:58.826087Z",
     "iopub.status.idle": "2026-03-12T18:53:58.832315Z",
     "shell.execute_reply": "2026-03-12T18:53:58.831731Z",
     "shell.execute_reply.started": "2026-03-12T18:53:58.826427Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def predict_bilstm(text):\n",
    "\n",
    "    enc = bilstm_tokenizer(\n",
    "        text,\n",
    "        truncation=True,\n",
    "        padding=True,\n",
    "        max_length=128,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "\n",
    "    enc = {k:v.to(DEVICE) for k,v in enc.items() if k != \"token_type_ids\"}\n",
    "\n",
    "    with torch.no_grad():\n",
    "\n",
    "        logits = bilstm_model(\n",
    "            enc[\"input_ids\"],\n",
    "            enc[\"attention_mask\"]\n",
    "        )\n",
    "\n",
    "        probs = torch.softmax(logits,dim=1)\n",
    "\n",
    "    probs = probs.cpu().numpy()[0]\n",
    "\n",
    "    labels = [\"Negative\",\"Neutral\",\"Positive\"]\n",
    "\n",
    "    pred = probs.argmax()\n",
    "\n",
    "    print(\"Input:\", text)\n",
    "    print(\"Prediction:\", labels[pred])\n",
    "\n",
    "    for i,p in enumerate(probs):\n",
    "        print(labels[i],\":\",round(float(p),4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T18:54:07.781584Z",
     "iopub.status.busy": "2026-03-12T18:54:07.780990Z",
     "iopub.status.idle": "2026-03-12T18:54:07.795695Z",
     "shell.execute_reply": "2026-03-12T18:54:07.794905Z",
     "shell.execute_reply.started": "2026-03-12T18:54:07.781547Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input: ఈ సినిమా చాలా బాగుంది\n",
      "Prediction: Positive\n",
      "Negative : 0.0725\n",
      "Neutral : 0.0815\n",
      "Positive : 0.846\n",
      "Input: ఈ సినిమా చాలా చెడ్డది\n",
      "Prediction: Neutral\n",
      "Negative : 0.3011\n",
      "Neutral : 0.4583\n",
      "Positive : 0.2406\n"
     ]
    }
   ],
   "source": [
    "predict_bilstm(\"ఈ సినిమా చాలా బాగుంది\")\n",
    "predict_bilstm(\"ఈ సినిమా చాలా చెడ్డది\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-12T19:16:27.947671Z",
     "iopub.status.busy": "2026-03-12T19:16:27.946777Z",
     "iopub.status.idle": "2026-03-12T19:16:34.707203Z",
     "shell.execute_reply": "2026-03-12T19:16:34.706625Z",
     "shell.execute_reply.started": "2026-03-12T19:16:27.947639Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading teacher...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-multilingual-cased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Teacher loaded correctly\n",
      "Loading DistilBERT student...\n",
      "DistilBERT loaded correctly\n",
      "Loading BiLSTM...\n",
      "BiLSTM loaded\n",
      "* Running on local URL:  http://127.0.0.1:7865\n",
      "* Running on public URL: https://10494463577ae56ce2.gradio.live\n",
      "\n",
      "This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"https://10494463577ae56ce2.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/gradio/queueing.py\", line 759, in process_events\n",
      "    response = await route_utils.call_process_api(\n",
      "               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/gradio/route_utils.py\", line 354, in call_process_api\n",
      "    output = await app.get_blocks().process_api(\n",
      "             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/gradio/blocks.py\", line 2116, in process_api\n",
      "    result = await self.call_function(\n",
      "             ^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/gradio/blocks.py\", line 1623, in call_function\n",
      "    prediction = await anyio.to_thread.run_sync(  # type: ignore\n",
      "                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/anyio/to_thread.py\", line 63, in run_sync\n",
      "    return await get_async_backend().run_sync_in_worker_thread(\n",
      "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/anyio/_backends/_asyncio.py\", line 2502, in run_sync_in_worker_thread\n",
      "    return await future\n",
      "           ^^^^^^^^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/anyio/_backends/_asyncio.py\", line 986, in run\n",
      "    result = context.run(func, *args)\n",
      "             ^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/gradio/utils.py\", line 915, in wrapper\n",
      "    response = f(*args, **kwargs)\n",
      "               ^^^^^^^^^^^^^^^^^^\n",
      "  File \"/tmp/ipykernel_55/2323462450.py\", line 227, in run_selected_model\n",
      "    fig = make_prob_chart(probs_to_dict(probs), f\"{model_name} Probabilities\")\n",
      "          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/tmp/ipykernel_55/140380090.py\", line 266, in make_prob_chart\n",
      "    ax.bar(LABELS,probs)\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/matplotlib/__init__.py\", line 1521, in inner\n",
      "    return func(\n",
      "           ^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/matplotlib/axes/_axes.py\", line 2544, in bar\n",
      "    self._process_unit_info(\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/matplotlib/axes/_base.py\", line 2617, in _process_unit_info\n",
      "    axis.update_units(data)\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/matplotlib/axis.py\", line 1765, in update_units\n",
      "    default = self._converter.default_units(data, self)\n",
      "              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/matplotlib/category.py\", line 106, in default_units\n",
      "    axis.set_units(UnitData(data))\n",
      "                   ^^^^^^^^^^^^^^\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/matplotlib/category.py\", line 182, in __init__\n",
      "    self.update(data)\n",
      "  File \"/usr/local/lib/python3.12/dist-packages/matplotlib/category.py\", line 215, in update\n",
      "    for val in OrderedDict.fromkeys(data):\n",
      "               ^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "TypeError: unhashable type: 'dict'\n"
     ]
    }
   ],
   "source": [
    "import time\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import gradio as gr\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from transformers import (\n",
    "    AutoTokenizer,\n",
    "    AutoModel,\n",
    "    AutoModelForSequenceClassification\n",
    ")\n",
    "\n",
    "from peft import LoraConfig, get_peft_model, TaskType\n",
    "\n",
    "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "LABELS = [\"Negative\",\"Neutral\",\"Positive\"]\n",
    "\n",
    "# =========================================\n",
    "# TOKENIZERS\n",
    "# =========================================\n",
    "\n",
    "muril_tokenizer = AutoTokenizer.from_pretrained(\"google/muril-base-cased\")\n",
    "\n",
    "student_tokenizer = AutoTokenizer.from_pretrained(\n",
    "    \"distilbert-base-multilingual-cased\"\n",
    ")\n",
    "\n",
    "# =========================================\n",
    "# TEACHER MODEL\n",
    "# =========================================\n",
    "\n",
    "class MurilPEFTModel(nn.Module):\n",
    "\n",
    "    def __init__(self, base_model, num_classes=3):\n",
    "\n",
    "        super().__init__()\n",
    "\n",
    "        self.base_model = base_model\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.classifier = nn.Linear(768, num_classes)\n",
    "\n",
    "    def mean_pool(self, output, mask):\n",
    "\n",
    "        token_embeddings = output.last_hidden_state\n",
    "\n",
    "        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()\n",
    "\n",
    "        summed = torch.sum(token_embeddings * mask, 1)\n",
    "\n",
    "        counts = torch.clamp(mask.sum(1), min=1e-9)\n",
    "\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "\n",
    "        output = self.base_model(\n",
    "            input_ids=input_ids,\n",
    "            attention_mask=attention_mask\n",
    "        )\n",
    "\n",
    "        pooled = self.mean_pool(output, attention_mask)\n",
    "\n",
    "        logits = self.classifier(self.dropout(pooled))\n",
    "\n",
    "        return logits\n",
    "\n",
    "\n",
    "print(\"Loading teacher...\")\n",
    "\n",
    "base_model = AutoModel.from_pretrained(\"google/muril-base-cased\")\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    target_modules=[\"query\",\"key\",\"value\",\"dense\"],\n",
    "    lora_dropout=0.1,\n",
    "    bias=\"none\",\n",
    "    task_type=TaskType.FEATURE_EXTRACTION\n",
    ")\n",
    "\n",
    "base_model = get_peft_model(base_model, lora_config)\n",
    "\n",
    "teacher_model = MurilPEFTModel(base_model).to(DEVICE)\n",
    "\n",
    "state = torch.load(\n",
    "    \"best_muril_lora.pt\",\n",
    "    map_location=DEVICE\n",
    ")\n",
    "\n",
    "teacher_model.load_state_dict(state)\n",
    "\n",
    "teacher_model.eval()\n",
    "\n",
    "print(\"Teacher loaded correctly\")\n",
    "\n",
    "# =========================================\n",
    "# DISTILBERT STUDENT\n",
    "# =========================================\n",
    "\n",
    "print(\"Loading DistilBERT student...\")\n",
    "\n",
    "distil_model = AutoModelForSequenceClassification.from_pretrained(\n",
    "    \"distilbert-base-multilingual-cased\",\n",
    "    num_labels=3\n",
    ")\n",
    "\n",
    "state = torch.load(\n",
    "    \"best_student.pt\",\n",
    "    map_location=DEVICE\n",
    ")\n",
    "\n",
    "distil_model.load_state_dict(state)\n",
    "\n",
    "distil_model.to(DEVICE)\n",
    "distil_model.eval()\n",
    "\n",
    "print(\"DistilBERT loaded correctly\")\n",
    "\n",
    "# =========================================\n",
    "# BILSTM STUDENT\n",
    "# =========================================\n",
    "\n",
    "class BiLSTMStudent(nn.Module):\n",
    "\n",
    "    def __init__(self, vocab_size, num_classes=3):\n",
    "\n",
    "        super().__init__()\n",
    "\n",
    "        self.embedding = nn.Embedding(vocab_size, 300)\n",
    "\n",
    "        self.lstm = nn.LSTM(\n",
    "            input_size=300,\n",
    "            hidden_size=256,\n",
    "            batch_first=True,\n",
    "            bidirectional=True\n",
    "        )\n",
    "\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "\n",
    "        self.fc = nn.Linear(512, num_classes)\n",
    "\n",
    "    def mean_pool(self, outputs, mask):\n",
    "\n",
    "        mask = mask.unsqueeze(-1)\n",
    "\n",
    "        summed = torch.sum(outputs * mask, dim=1)\n",
    "\n",
    "        counts = torch.clamp(mask.sum(dim=1), min=1e-9)\n",
    "\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "\n",
    "        x = self.embedding(input_ids)\n",
    "\n",
    "        outputs, _ = self.lstm(x)\n",
    "\n",
    "        pooled = self.mean_pool(outputs, attention_mask)\n",
    "\n",
    "        logits = self.fc(self.dropout(pooled))\n",
    "\n",
    "        return logits\n",
    "\n",
    "\n",
    "print(\"Loading BiLSTM...\")\n",
    "\n",
    "bilstm_model = BiLSTMStudent(muril_tokenizer.vocab_size).to(DEVICE)\n",
    "\n",
    "bilstm_model.load_state_dict(\n",
    "    torch.load(\n",
    "        \"best_bilstm_offline_kd.pt\",\n",
    "        map_location=DEVICE\n",
    "    )\n",
    ")\n",
    "\n",
    "bilstm_model.eval()\n",
    "\n",
    "print(\"BiLSTM loaded\")\n",
    "\n",
    "# =========================================\n",
    "# PREDICTION FUNCTIONS\n",
    "# =========================================\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_teacher(text):\n",
    "\n",
    "    enc = muril_tokenizer(\n",
    "        text,\n",
    "        padding=\"max_length\",\n",
    "        truncation=True,\n",
    "        max_length=128,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "\n",
    "    enc = {k:v.to(DEVICE) for k,v in enc.items() if k!=\"token_type_ids\"}\n",
    "\n",
    "    start=time.time()\n",
    "\n",
    "    logits = teacher_model(**enc)\n",
    "\n",
    "    latency=(time.time()-start)*1000\n",
    "\n",
    "    probs=torch.softmax(logits,dim=1).cpu().numpy()[0]\n",
    "\n",
    "    return probs,latency\n",
    "\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_distilbert(text):\n",
    "\n",
    "    enc = student_tokenizer(\n",
    "        text,\n",
    "        truncation=True,\n",
    "        padding=True,\n",
    "        max_length=128,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "\n",
    "    enc = {k:v.to(DEVICE) for k,v in enc.items() if k!=\"token_type_ids\"}\n",
    "\n",
    "    start=time.time()\n",
    "\n",
    "    outputs = distil_model(**enc)\n",
    "\n",
    "    latency=(time.time()-start)*1000\n",
    "\n",
    "    probs=torch.softmax(outputs.logits,dim=1).cpu().numpy()[0]\n",
    "\n",
    "    return probs,latency\n",
    "\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_bilstm(text):\n",
    "\n",
    "    enc = muril_tokenizer(\n",
    "        text,\n",
    "        truncation=True,\n",
    "        padding=True,\n",
    "        max_length=128,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "\n",
    "    enc = {k:v.to(DEVICE) for k,v in enc.items() if k!=\"token_type_ids\"}\n",
    "\n",
    "    start=time.time()\n",
    "\n",
    "    logits=bilstm_model(enc[\"input_ids\"],enc[\"attention_mask\"])\n",
    "\n",
    "    latency=(time.time()-start)*1000\n",
    "\n",
    "    probs=torch.softmax(logits,dim=1).cpu().numpy()[0]\n",
    "\n",
    "    return probs,latency\n",
    "\n",
    "# =========================================\n",
    "# CHARTS\n",
    "# =========================================\n",
    "\n",
    "def make_prob_chart(probs,title):\n",
    "\n",
    "    fig,ax=plt.subplots(figsize=(6,4))\n",
    "\n",
    "    ax.bar(LABELS,probs)\n",
    "\n",
    "    ax.set_ylim(0,1)\n",
    "\n",
    "    ax.set_ylabel(\"Probability\")\n",
    "\n",
    "    ax.set_title(title)\n",
    "\n",
    "    return fig\n",
    "\n",
    "\n",
    "def make_comparison_chart(tp,dp,bp):\n",
    "\n",
    "    labels=[\"Negative\",\"Neutral\",\"Positive\"]\n",
    "\n",
    "    x=list(range(3))\n",
    "    width=0.25\n",
    "\n",
    "    fig,ax=plt.subplots(figsize=(8,4))\n",
    "\n",
    "    ax.bar([i-width for i in x],tp,width,label=\"Teacher\")\n",
    "    ax.bar(x,dp,width,label=\"DistilBERT\")\n",
    "    ax.bar([i+width for i in x],bp,width,label=\"BiLSTM\")\n",
    "\n",
    "    ax.set_xticks(x)\n",
    "    ax.set_xticklabels(labels)\n",
    "\n",
    "    ax.set_ylim(0,1)\n",
    "\n",
    "    ax.legend()\n",
    "\n",
    "    ax.set_title(\"Model Comparison\")\n",
    "\n",
    "    return fig\n",
    "\n",
    "# =========================================\n",
    "# UI FUNCTIONS\n",
    "# =========================================\n",
    "\n",
    "def run_selected_model(text,model,language):\n",
    "\n",
    "    if model==\"Teacher (MuRIL + LoRA)\":\n",
    "        probs,lat=predict_teacher(text)\n",
    "\n",
    "    elif model==\"Student (DistilBERT)\":\n",
    "        probs,lat=predict_distilbert(text)\n",
    "\n",
    "    else:\n",
    "        probs,lat=predict_bilstm(text)\n",
    "\n",
    "    pred=LABELS[int(probs.argmax())]\n",
    "\n",
    "    result=f\"\"\"\n",
    "Model: {model}\n",
    "\n",
    "Prediction: {pred}\n",
    "\n",
    "Confidence: {probs.max():.4f}\n",
    "\n",
    "Latency: {lat:.2f} ms\n",
    "\"\"\"\n",
    "\n",
    "    fig=make_prob_chart(probs,f\"{model} Probabilities\")\n",
    "\n",
    "    return result,fig\n",
    "\n",
    "\n",
    "def compare_all_models(text,language):\n",
    "\n",
    "    tp,t1=predict_teacher(text)\n",
    "    dp,t2=predict_distilbert(text)\n",
    "    bp,t3=predict_bilstm(text)\n",
    "\n",
    "    rows=[]\n",
    "\n",
    "    for name,p,t in [\n",
    "        (\"Teacher\",tp,t1),\n",
    "        (\"DistilBERT\",dp,t2),\n",
    "        (\"BiLSTM\",bp,t3)\n",
    "    ]:\n",
    "\n",
    "        rows.append({\n",
    "            \"Model\":name,\n",
    "            \"Prediction\":LABELS[int(p.argmax())],\n",
    "            \"Confidence\":round(float(p.max()),4),\n",
    "            \"Negative\":round(float(p[0]),4),\n",
    "            \"Neutral\":round(float(p[1]),4),\n",
    "            \"Positive\":round(float(p[2]),4),\n",
    "            \"Latency(ms)\":round(float(t),2)\n",
    "        })\n",
    "\n",
    "    df=pd.DataFrame(rows)\n",
    "\n",
    "    fig=make_comparison_chart(tp,dp,bp)\n",
    "\n",
    "    return df,fig\n",
    "\n",
    "# =========================================\n",
    "# GRADIO UI\n",
    "# =========================================\n",
    "\n",
    "with gr.Blocks(title=\"Indic Multilingual Sentiment Analysis Demo\") as demo:\n",
    "\n",
    "    gr.Markdown(\"# Low-Cost Multilingual Sentiment Analysis\")\n",
    "\n",
    "    gr.Markdown(\n",
    "        \"Teacher: MuRIL + LoRA | Students: DistilBERT and BiLSTM\"\n",
    "    )\n",
    "\n",
    "    with gr.Row():\n",
    "\n",
    "        with gr.Column():\n",
    "\n",
    "            text_input=gr.Textbox(\n",
    "                label=\"Enter text\",\n",
    "                lines=4\n",
    "            )\n",
    "\n",
    "            model_choice=gr.Dropdown(\n",
    "                [\"Teacher (MuRIL + LoRA)\",\n",
    "                 \"Student (DistilBERT)\",\n",
    "                 \"Student (BiLSTM)\"],\n",
    "                value=\"Teacher (MuRIL + LoRA)\"\n",
    "            )\n",
    "\n",
    "            language_choice=gr.Dropdown(\n",
    "                [\"Telugu\",\"Kannada\",\"Marathi\"],\n",
    "                value=\"Telugu\"\n",
    "            )\n",
    "\n",
    "            predict_btn=gr.Button(\"Predict\")\n",
    "\n",
    "            compare_btn=gr.Button(\"Compare Models\")\n",
    "\n",
    "        with gr.Column():\n",
    "\n",
    "            result_output=gr.Markdown()\n",
    "\n",
    "            chart_output=gr.Plot()\n",
    "\n",
    "    gr.Markdown(\"## Model Comparison\")\n",
    "\n",
    "    comparison_table=gr.Dataframe()\n",
    "\n",
    "    comparison_chart=gr.Plot()\n",
    "\n",
    "    predict_btn.click(\n",
    "        fn=run_selected_model,\n",
    "        inputs=[text_input,model_choice,language_choice],\n",
    "        outputs=[result_output,chart_output]\n",
    "    )\n",
    "\n",
    "    compare_btn.click(\n",
    "        fn=compare_all_models,\n",
    "        inputs=[text_input,language_choice],\n",
    "        outputs=[comparison_table,comparison_chart]\n",
    "    )\n",
    "\n",
    "demo.launch()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-18T08:36:38.961129Z",
     "iopub.status.busy": "2026-03-18T08:36:38.960729Z",
     "iopub.status.idle": "2026-03-18T08:37:29.229180Z",
     "shell.execute_reply": "2026-03-18T08:37:29.228313Z",
     "shell.execute_reply.started": "2026-03-18T08:36:38.961098Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-03-18 08:36:58.520638: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
      "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
      "E0000 00:00:1773823018.720257      55 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
      "E0000 00:00:1773823018.776504      55 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
      "W0000 00:00:1773823019.250188      55 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "W0000 00:00:1773823019.250222      55 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "W0000 00:00:1773823019.250224      55 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "W0000 00:00:1773823019.250227      55 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a25c78072d314134b27b4b6cb43d1df2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/206 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f6fc2a8c116d491d87e99cccf73cad50",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/411 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a9c3d6ffe3194525b5f156b520fc77cd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "vocab.txt: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "735ff9929ad9417ab2514bddb0c8875b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "special_tokens_map.json:   0%|          | 0.00/113 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "45450327de284745bd01d41b02760cd9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a8a5a92a92fe49e8a44f90a2214f0d1f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/466 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fa36008856ff48f2aed834a33f837c1a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "vocab.txt: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f306e173122d4ca18c51bf6a7a839714",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer.json: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading Teacher...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "150899beb84e482eb54fc38060464598",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "pytorch_model.bin:   0%|          | 0.00/953M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e92e44a012e44bfca8a49c12e2bb18ea",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/953M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Teacher loaded\n",
      "Loading DistilBERT...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c25ab0d5164d4793bfc426afc5949290",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/542M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-multilingual-cased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DistilBERT loaded\n",
      "Loading BiLSTM...\n",
      "BiLSTM loaded\n",
      "* Running on local URL:  http://127.0.0.1:7860\n",
      "* Running on public URL: https://403499a212981be455.gradio.live\n",
      "\n",
      "This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"https://403499a212981be455.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import time\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import gradio as gr\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from transformers import (\n",
    "    AutoTokenizer,\n",
    "    AutoModel,\n",
    "    AutoModelForSequenceClassification\n",
    ")\n",
    "\n",
    "from peft import LoraConfig, get_peft_model, TaskType\n",
    "\n",
    "# =========================\n",
    "# CONFIG\n",
    "# =========================\n",
    "\n",
    "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "MAX_LEN = 128\n",
    "NUM_CLASSES = 3\n",
    "\n",
    "LABEL_MAP = {\n",
    "    0: \"Negative\",\n",
    "    1: \"Neutral\",\n",
    "    2: \"Positive\"\n",
    "}\n",
    "\n",
    "# =========================\n",
    "# TOKENIZERS\n",
    "# =========================\n",
    "\n",
    "muril_tokenizer = AutoTokenizer.from_pretrained(\"google/muril-base-cased\")\n",
    "\n",
    "student_tokenizer = AutoTokenizer.from_pretrained(\n",
    "    \"distilbert-base-multilingual-cased\"\n",
    ")\n",
    "\n",
    "# =========================\n",
    "# TEACHER MODEL\n",
    "# =========================\n",
    "\n",
    "class MurilPEFTModel(nn.Module):\n",
    "\n",
    "    def __init__(self, base_model, num_classes=3):\n",
    "\n",
    "        super().__init__()\n",
    "\n",
    "        self.base_model = base_model\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.classifier = nn.Linear(768, num_classes)\n",
    "\n",
    "    def mean_pool(self, output, mask):\n",
    "\n",
    "        token_embeddings = output.last_hidden_state\n",
    "\n",
    "        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()\n",
    "\n",
    "        summed = torch.sum(token_embeddings * mask, 1)\n",
    "\n",
    "        counts = torch.clamp(mask.sum(1), min=1e-9)\n",
    "\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "\n",
    "        output = self.base_model(\n",
    "            input_ids=input_ids,\n",
    "            attention_mask=attention_mask\n",
    "        )\n",
    "\n",
    "        pooled = self.mean_pool(output, attention_mask)\n",
    "\n",
    "        logits = self.classifier(self.dropout(pooled))\n",
    "\n",
    "        return logits\n",
    "\n",
    "\n",
    "print(\"Loading Teacher...\")\n",
    "\n",
    "base_model = AutoModel.from_pretrained(\"google/muril-base-cased\")\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    target_modules=[\"query\",\"key\",\"value\",\"dense\"],\n",
    "    lora_dropout=0.1,\n",
    "    bias=\"none\",\n",
    "    task_type=TaskType.FEATURE_EXTRACTION\n",
    ")\n",
    "\n",
    "base_model = get_peft_model(base_model, lora_config)\n",
    "\n",
    "teacher_model = MurilPEFTModel(base_model).to(DEVICE)\n",
    "\n",
    "state = torch.load(\n",
    "    \"best_muril_lora.pt\",\n",
    "    map_location=DEVICE\n",
    ")\n",
    "\n",
    "teacher_model.load_state_dict(state)\n",
    "\n",
    "teacher_model.eval()\n",
    "\n",
    "print(\"Teacher loaded\")\n",
    "\n",
    "# =========================\n",
    "# DISTILBERT\n",
    "# =========================\n",
    "\n",
    "print(\"Loading DistilBERT...\")\n",
    "\n",
    "distil_model = AutoModelForSequenceClassification.from_pretrained(\n",
    "    \"distilbert-base-multilingual-cased\",\n",
    "    num_labels=3\n",
    ")\n",
    "\n",
    "distil_model.load_state_dict(\n",
    "    torch.load(\"best_student.pt\", map_location=DEVICE)\n",
    ")\n",
    "\n",
    "distil_model.to(DEVICE)\n",
    "distil_model.eval()\n",
    "\n",
    "print(\"DistilBERT loaded\")\n",
    "\n",
    "# =========================\n",
    "# BILSTM\n",
    "# =========================\n",
    "\n",
    "class BiLSTMStudent(nn.Module):\n",
    "\n",
    "    def __init__(self, vocab_size, num_classes=3):\n",
    "\n",
    "        super().__init__()\n",
    "\n",
    "        self.embedding = nn.Embedding(vocab_size, 300)\n",
    "\n",
    "        self.lstm = nn.LSTM(\n",
    "            input_size=300,\n",
    "            hidden_size=256,\n",
    "            batch_first=True,\n",
    "            bidirectional=True\n",
    "        )\n",
    "\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "\n",
    "        self.fc = nn.Linear(512, num_classes)\n",
    "\n",
    "    def mean_pool(self, outputs, mask):\n",
    "\n",
    "        mask = mask.unsqueeze(-1)\n",
    "\n",
    "        summed = torch.sum(outputs * mask, dim=1)\n",
    "\n",
    "        counts = torch.clamp(mask.sum(dim=1), min=1e-9)\n",
    "\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "\n",
    "        x = self.embedding(input_ids)\n",
    "\n",
    "        outputs, _ = self.lstm(x)\n",
    "\n",
    "        pooled = self.mean_pool(outputs, attention_mask)\n",
    "\n",
    "        logits = self.fc(self.dropout(pooled))\n",
    "\n",
    "        return logits\n",
    "\n",
    "\n",
    "print(\"Loading BiLSTM...\")\n",
    "\n",
    "bilstm_model = BiLSTMStudent(muril_tokenizer.vocab_size).to(DEVICE)\n",
    "\n",
    "bilstm_model.load_state_dict(\n",
    "    torch.load(\n",
    "        \"best_bilstm_offline_kd.pt\",\n",
    "        map_location=DEVICE\n",
    "    )\n",
    ")\n",
    "\n",
    "bilstm_model.eval()\n",
    "\n",
    "print(\"BiLSTM loaded\")\n",
    "\n",
    "# =========================\n",
    "# HELPERS\n",
    "# =========================\n",
    "\n",
    "def encode_text(text, tokenizer):\n",
    "\n",
    "    enc = tokenizer(\n",
    "        text,\n",
    "        padding=\"max_length\",\n",
    "        truncation=True,\n",
    "        max_length=MAX_LEN,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "\n",
    "    return {k:v.to(DEVICE) for k,v in enc.items() if k!=\"token_type_ids\"}\n",
    "\n",
    "\n",
    "def probs_to_dict(probs):\n",
    "\n",
    "    return {\n",
    "        \"Negative\": float(probs[0]),\n",
    "        \"Neutral\": float(probs[1]),\n",
    "        \"Positive\": float(probs[2]),\n",
    "    }\n",
    "\n",
    "\n",
    "def format_single_result(model_name, probs, elapsed_ms):\n",
    "\n",
    "    pred_idx = int(probs.argmax())\n",
    "\n",
    "    pred_label = LABEL_MAP[pred_idx]\n",
    "\n",
    "    confidence = float(probs[pred_idx])\n",
    "\n",
    "    return (\n",
    "        f\"**Model:** {model_name}\\n\\n\"\n",
    "        f\"**Prediction:** {pred_label}\\n\\n\"\n",
    "        f\"**Confidence:** {confidence:.4f}\\n\\n\"\n",
    "        f\"**Latency:** {elapsed_ms:.2f} ms\"\n",
    "    )\n",
    "\n",
    "# =========================\n",
    "# PREDICTION\n",
    "# =========================\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_teacher(text):\n",
    "\n",
    "    inputs = encode_text(text, muril_tokenizer)\n",
    "\n",
    "    start = time.time()\n",
    "\n",
    "    logits = teacher_model(**inputs)\n",
    "\n",
    "    elapsed = (time.time()-start)*1000\n",
    "\n",
    "    probs = F.softmax(logits, dim=1).cpu().numpy()[0]\n",
    "\n",
    "    return probs, elapsed\n",
    "\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_distilbert(text):\n",
    "\n",
    "    inputs = encode_text(text, student_tokenizer)\n",
    "\n",
    "    start = time.time()\n",
    "\n",
    "    outputs = distil_model(**inputs)\n",
    "\n",
    "    elapsed = (time.time()-start)*1000\n",
    "\n",
    "    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]\n",
    "\n",
    "    return probs, elapsed\n",
    "\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_bilstm(text):\n",
    "\n",
    "    inputs = encode_text(text, muril_tokenizer)\n",
    "\n",
    "    start = time.time()\n",
    "\n",
    "    logits = bilstm_model(\n",
    "        inputs[\"input_ids\"],\n",
    "        inputs[\"attention_mask\"]\n",
    "    )\n",
    "\n",
    "    elapsed = (time.time()-start)*1000\n",
    "\n",
    "    probs = F.softmax(logits, dim=1).cpu().numpy()[0]\n",
    "\n",
    "    return probs, elapsed\n",
    "\n",
    "# =========================\n",
    "# CHARTS\n",
    "# =========================\n",
    "\n",
    "def make_prob_chart(prob_dict, title):\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(6,4))\n",
    "\n",
    "    ax.bar(list(prob_dict.keys()), list(prob_dict.values()))\n",
    "\n",
    "    ax.set_ylim(0,1)\n",
    "\n",
    "    ax.set_ylabel(\"Probability\")\n",
    "\n",
    "    ax.set_title(title)\n",
    "\n",
    "    plt.tight_layout()\n",
    "\n",
    "    return fig\n",
    "\n",
    "\n",
    "def make_comparison_chart(t,d,b):\n",
    "\n",
    "    labels=[\"Negative\",\"Neutral\",\"Positive\"]\n",
    "\n",
    "    x=list(range(3))\n",
    "    width=0.25\n",
    "\n",
    "    fig,ax=plt.subplots(figsize=(8,4))\n",
    "\n",
    "    ax.bar([i-width for i in x],t,width,label=\"Teacher\")\n",
    "    ax.bar(x,d,width,label=\"DistilBERT\")\n",
    "    ax.bar([i+width for i in x],b,width,label=\"BiLSTM\")\n",
    "\n",
    "    ax.set_xticks(x)\n",
    "    ax.set_xticklabels(labels)\n",
    "\n",
    "    ax.set_ylim(0,1)\n",
    "\n",
    "    ax.set_ylabel(\"Probability\")\n",
    "\n",
    "    ax.legend()\n",
    "\n",
    "    ax.set_title(\"Model Comparison\")\n",
    "\n",
    "    plt.tight_layout()\n",
    "\n",
    "    return fig\n",
    "\n",
    "# =========================\n",
    "# UI FUNCTIONS\n",
    "# =========================\n",
    "\n",
    "def run_selected_model(text, model_name, language):\n",
    "\n",
    "    if model_name==\"Teacher (MuRIL + LoRA)\":\n",
    "        probs,elapsed=predict_teacher(text)\n",
    "\n",
    "    elif model_name==\"Student (DistilBERT)\":\n",
    "        probs,elapsed=predict_distilbert(text)\n",
    "\n",
    "    else:\n",
    "        probs,elapsed=predict_bilstm(text)\n",
    "\n",
    "    result=format_single_result(model_name,probs,elapsed)\n",
    "\n",
    "    fig=make_prob_chart(probs_to_dict(probs),f\"{model_name} Probabilities\")\n",
    "\n",
    "    return result,fig\n",
    "\n",
    "\n",
    "def compare_all_models(text,language):\n",
    "\n",
    "    tp,t1=predict_teacher(text)\n",
    "    dp,t2=predict_distilbert(text)\n",
    "    bp,t3=predict_bilstm(text)\n",
    "\n",
    "    rows=[]\n",
    "\n",
    "    for name,p,t in [\n",
    "\n",
    "        (\"Teacher (MuRIL + LoRA)\",tp,t1),\n",
    "        (\"Student (DistilBERT)\",dp,t2),\n",
    "        (\"Student (BiLSTM)\",bp,t3)\n",
    "\n",
    "    ]:\n",
    "\n",
    "        pred_idx=int(p.argmax())\n",
    "\n",
    "        rows.append({\n",
    "\n",
    "            \"Model\":name,\n",
    "            \"Prediction\":LABEL_MAP[pred_idx],\n",
    "            \"Confidence\":round(float(p[pred_idx]),4),\n",
    "            \"Negative\":round(float(p[0]),4),\n",
    "            \"Neutral\":round(float(p[1]),4),\n",
    "            \"Positive\":round(float(p[2]),4),\n",
    "            \"Latency (ms)\":round(float(t),2)\n",
    "\n",
    "        })\n",
    "\n",
    "    df=pd.DataFrame(rows)\n",
    "\n",
    "    fig=make_comparison_chart(tp,dp,bp)\n",
    "\n",
    "    return df,fig\n",
    "\n",
    "# =========================\n",
    "# EXAMPLES\n",
    "# =========================\n",
    "\n",
    "examples=[\n",
    "[\"ఈ సినిమా చాలా బాగుంది\",\"Teacher (MuRIL + LoRA)\",\"Telugu\"],\n",
    "[\"ಈ ಸಿನಿಮಾ ಚೆನ್ನಾಗಿದೆ\",\"Student (DistilBERT)\",\"Kannada\"],\n",
    "[\"हा चित्रपट खूप छान आहे\",\"Student (BiLSTM)\",\"Marathi\"]\n",
    "]\n",
    "\n",
    "# =========================\n",
    "# GRADIO UI\n",
    "# =========================\n",
    "\n",
    "with gr.Blocks(title=\"Indic Multilingual Sentiment Analysis Demo\") as demo:\n",
    "\n",
    "    gr.Markdown(\"# Low-Cost System for Multilingual Sentiment Analysis in Indic Languages\")\n",
    "\n",
    "    gr.Markdown(\n",
    "        \"Train on **Telugu**, test/demo on **Telugu, Kannada, Marathi**.\\n\"\n",
    "        \"Models: **MuRIL + LoRA Teacher**, **DistilBERT Student**, **BiLSTM Student**.\"\n",
    "    )\n",
    "\n",
    "    with gr.Row():\n",
    "\n",
    "        with gr.Column():\n",
    "\n",
    "            text_input=gr.Textbox(\n",
    "                label=\"Enter text\",\n",
    "                lines=4,\n",
    "                placeholder=\"Type Telugu / Kannada / Marathi text here...\"\n",
    "            )\n",
    "\n",
    "            model_choice=gr.Dropdown(\n",
    "                [\"Teacher (MuRIL + LoRA)\",\n",
    "                 \"Student (DistilBERT)\",\n",
    "                 \"Student (BiLSTM)\"],\n",
    "                value=\"Teacher (MuRIL + LoRA)\",\n",
    "                label=\"Choose model\"\n",
    "            )\n",
    "\n",
    "            language_choice=gr.Dropdown(\n",
    "                [\"Telugu\",\"Kannada\",\"Marathi\"],\n",
    "                value=\"Telugu\",\n",
    "                label=\"Language\"\n",
    "            )\n",
    "\n",
    "            with gr.Row():\n",
    "\n",
    "                predict_btn=gr.Button(\"Predict\")\n",
    "\n",
    "                compare_btn=gr.Button(\"Compare All Models\")\n",
    "\n",
    "        with gr.Column():\n",
    "\n",
    "            result_output=gr.Markdown()\n",
    "\n",
    "            chart_output=gr.Plot()\n",
    "\n",
    "    gr.Markdown(\"## Model Comparison\")\n",
    "\n",
    "    comparison_table=gr.Dataframe()\n",
    "\n",
    "    comparison_chart=gr.Plot()\n",
    "\n",
    "    gr.Examples(\n",
    "        examples=examples,\n",
    "        inputs=[text_input,model_choice,language_choice]\n",
    "    )\n",
    "\n",
    "    predict_btn.click(\n",
    "        fn=run_selected_model,\n",
    "        inputs=[text_input,model_choice,language_choice],\n",
    "        outputs=[result_output,chart_output]\n",
    "    )\n",
    "\n",
    "    compare_btn.click(\n",
    "        fn=compare_all_models,\n",
    "        inputs=[text_input,language_choice],\n",
    "        outputs=[comparison_table,comparison_chart]\n",
    "    )\n",
    "\n",
    "demo.launch()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-21T17:05:19.081989Z",
     "iopub.status.busy": "2026-04-21T17:05:19.081362Z",
     "iopub.status.idle": "2026-04-21T17:06:22.275255Z",
     "shell.execute_reply": "2026-04-21T17:06:22.274423Z",
     "shell.execute_reply.started": "2026-04-21T17:05:19.081956Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-21 17:05:45.890505: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
      "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
      "E0000 00:00:1776791146.223087      55 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
      "E0000 00:00:1776791146.341330      55 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
      "W0000 00:00:1776791147.227156      55 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "W0000 00:00:1776791147.227193      55 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "W0000 00:00:1776791147.227196      55 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n",
      "W0000 00:00:1776791147.227199      55 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2f7807c3185d42cdaaecdded3fc5c1b7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/206 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "be243254472b411b8657f2a56f94f073",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/411 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5f878e52b01e4ed09e8350fcd53c8a45",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "vocab.txt: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3b6b919d0612414395663d68f3702324",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "special_tokens_map.json:   0%|          | 0.00/113 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f21f13f5d2d64345b88c25865fb10988",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "75669b4c874c4c2396385309841f3ed9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/466 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "67cf9261fe804a9bb564414648ee048e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "vocab.txt: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "02870445f37c4456818e82a31a02a48f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer.json: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading Teacher...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6ceb798ea07046d6a19ce89df68ab792",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "pytorch_model.bin:   0%|          | 0.00/953M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "99e7174a0df54f908bd5e9c414305cb5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/953M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Teacher loaded\n",
      "Loading DistilBERT...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e0fea034114842509abfbe88ac0fcc14",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/542M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-multilingual-cased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DistilBERT loaded\n",
      "Loading BiLSTM...\n",
      "BiLSTM loaded\n",
      "* Running on local URL:  http://127.0.0.1:7860\n",
      "* Running on public URL: https://84a8563f9ae4158619.gradio.live\n",
      "\n",
      "This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"https://84a8563f9ae4158619.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# =========================\n",
    "# IMPORTS\n",
    "# =========================\n",
    "\n",
    "import time\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import gradio as gr\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from transformers import (\n",
    "    AutoTokenizer,\n",
    "    AutoModel,\n",
    "    AutoModelForSequenceClassification\n",
    ")\n",
    "\n",
    "from peft import LoraConfig, get_peft_model, TaskType\n",
    "\n",
    "# =========================\n",
    "# CONFIG\n",
    "# =========================\n",
    "\n",
    "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "MAX_LEN = 128\n",
    "NUM_CLASSES = 3\n",
    "CONF_THRESHOLD = 0.6\n",
    "\n",
    "LABEL_MAP = {\n",
    "    0: \"Negative\",\n",
    "    1: \"Neutral\",\n",
    "    2: \"Positive\"\n",
    "}\n",
    "\n",
    "# =========================\n",
    "# TOKENIZERS\n",
    "# =========================\n",
    "\n",
    "muril_tokenizer = AutoTokenizer.from_pretrained(\"google/muril-base-cased\")\n",
    "\n",
    "student_tokenizer = AutoTokenizer.from_pretrained(\n",
    "    \"distilbert-base-multilingual-cased\"\n",
    ")\n",
    "\n",
    "# =========================\n",
    "# TEACHER MODEL\n",
    "# =========================\n",
    "\n",
    "class MurilPEFTModel(nn.Module):\n",
    "\n",
    "    def __init__(self, base_model, num_classes=3):\n",
    "        super().__init__()\n",
    "        self.base_model = base_model\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.classifier = nn.Linear(768, num_classes)\n",
    "\n",
    "    def mean_pool(self, output, mask):\n",
    "        token_embeddings = output.last_hidden_state\n",
    "        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()\n",
    "        summed = torch.sum(token_embeddings * mask, 1)\n",
    "        counts = torch.clamp(mask.sum(1), min=1e-9)\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)\n",
    "        pooled = self.mean_pool(output, attention_mask)\n",
    "        logits = self.classifier(self.dropout(pooled))\n",
    "        return logits\n",
    "\n",
    "\n",
    "print(\"Loading Teacher...\")\n",
    "\n",
    "base_model = AutoModel.from_pretrained(\"google/muril-base-cased\")\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    target_modules=[\"query\",\"key\",\"value\",\"dense\"],\n",
    "    lora_dropout=0.1,\n",
    "    bias=\"none\",\n",
    "    task_type=TaskType.FEATURE_EXTRACTION\n",
    ")\n",
    "\n",
    "base_model = get_peft_model(base_model, lora_config)\n",
    "\n",
    "teacher_model = MurilPEFTModel(base_model).to(DEVICE)\n",
    "\n",
    "teacher_model.load_state_dict(\n",
    "    torch.load(\"best_muril_lora.pt\", map_location=DEVICE)\n",
    ")\n",
    "\n",
    "teacher_model.eval()\n",
    "print(\"Teacher loaded\")\n",
    "\n",
    "# =========================\n",
    "# DISTILBERT\n",
    "# =========================\n",
    "\n",
    "print(\"Loading DistilBERT...\")\n",
    "\n",
    "distil_model = AutoModelForSequenceClassification.from_pretrained(\n",
    "    \"distilbert-base-multilingual-cased\",\n",
    "    num_labels=3\n",
    ")\n",
    "\n",
    "distil_model.load_state_dict(\n",
    "    torch.load(\"best_student.pt\", map_location=DEVICE)\n",
    ")\n",
    "\n",
    "distil_model.to(DEVICE)\n",
    "distil_model.eval()\n",
    "\n",
    "print(\"DistilBERT loaded\")\n",
    "\n",
    "# =========================\n",
    "# BILSTM\n",
    "# =========================\n",
    "\n",
    "class BiLSTMStudent(nn.Module):\n",
    "\n",
    "    def __init__(self, vocab_size, num_classes=3):\n",
    "        super().__init__()\n",
    "        self.embedding = nn.Embedding(vocab_size, 300)\n",
    "        self.lstm = nn.LSTM(300, 256, batch_first=True, bidirectional=True)\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.fc = nn.Linear(512, num_classes)\n",
    "\n",
    "    def mean_pool(self, outputs, mask):\n",
    "        mask = mask.unsqueeze(-1)\n",
    "        summed = torch.sum(outputs * mask, dim=1)\n",
    "        counts = torch.clamp(mask.sum(dim=1), min=1e-9)\n",
    "        return summed / counts\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "        x = self.embedding(input_ids)\n",
    "        outputs, _ = self.lstm(x)\n",
    "        pooled = self.mean_pool(outputs, attention_mask)\n",
    "        logits = self.fc(self.dropout(pooled))\n",
    "        return logits\n",
    "\n",
    "\n",
    "print(\"Loading BiLSTM...\")\n",
    "\n",
    "bilstm_model = BiLSTMStudent(muril_tokenizer.vocab_size).to(DEVICE)\n",
    "\n",
    "bilstm_model.load_state_dict(\n",
    "    torch.load(\"best_bilstm_offline_kd.pt\", map_location=DEVICE)\n",
    ")\n",
    "\n",
    "bilstm_model.eval()\n",
    "print(\"BiLSTM loaded\")\n",
    "\n",
    "# =========================\n",
    "# HELPERS\n",
    "# =========================\n",
    "\n",
    "def encode_text(text, tokenizer):\n",
    "    enc = tokenizer(\n",
    "        text,\n",
    "        padding=\"max_length\",\n",
    "        truncation=True,\n",
    "        max_length=MAX_LEN,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "    return {k:v.to(DEVICE) for k,v in enc.items() if k!=\"token_type_ids\"}\n",
    "\n",
    "\n",
    "def probs_to_dict(probs):\n",
    "    return {\n",
    "        \"Negative\": float(probs[0]),\n",
    "        \"Neutral\": float(probs[1]),\n",
    "        \"Positive\": float(probs[2]),\n",
    "    }\n",
    "\n",
    "# =========================\n",
    "# TRUST + EXPLAINABILITY\n",
    "# =========================\n",
    "\n",
    "def trust_decision(pred_label, confidence):\n",
    "    if confidence < CONF_THRESHOLD:\n",
    "        return \"⚠️ UNCERTAIN\", True\n",
    "    return pred_label, False\n",
    "\n",
    "\n",
    "def detect_failure(text, confidence):\n",
    "    if len(text.strip()) < 5:\n",
    "        return \"⚠️ Input too short\"\n",
    "    if confidence < 0.5:\n",
    "        return \"⚠️ Low confidence\"\n",
    "    return \"✅ OK\"\n",
    "\n",
    "\n",
    "def get_token_importance(text, model_fn):\n",
    "    words = text.split()[:6]\n",
    "\n",
    "    if len(words) <= 1:\n",
    "        return []\n",
    "\n",
    "    base_probs, _ = model_fn(text)\n",
    "    base_conf = float(max(base_probs))\n",
    "\n",
    "    scores = []\n",
    "    for i in range(len(words)):\n",
    "        new_text = \" \".join(words[:i] + words[i+1:])\n",
    "        new_probs, _ = model_fn(new_text)\n",
    "        new_conf = float(max(new_probs))\n",
    "        impact = base_conf - new_conf\n",
    "        scores.append((words[i], impact))\n",
    "\n",
    "    return sorted(scores, key=lambda x: x[1], reverse=True)\n",
    "\n",
    "\n",
    "def format_explanation(importances):\n",
    "    if not importances:\n",
    "        return \"No explanation\"\n",
    "\n",
    "    text = \"### 🔍 Important Words:\\n\\n\"\n",
    "    for w, s in importances:\n",
    "        text += f\"- **{w}** ({s:.3f})\\n\"\n",
    "\n",
    "    return text\n",
    "\n",
    "# =========================\n",
    "# PREDICTION\n",
    "# =========================\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_teacher(text):\n",
    "    inputs = encode_text(text, muril_tokenizer)\n",
    "    start = time.time()\n",
    "    logits = teacher_model(**inputs)\n",
    "    elapsed = (time.time()-start)*1000\n",
    "    probs = F.softmax(logits, dim=1).cpu().numpy()[0]\n",
    "    return probs, elapsed\n",
    "\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_distilbert(text):\n",
    "    inputs = encode_text(text, student_tokenizer)\n",
    "    start = time.time()\n",
    "    outputs = distil_model(**inputs)\n",
    "    elapsed = (time.time()-start)*1000\n",
    "    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]\n",
    "    return probs, elapsed\n",
    "\n",
    "\n",
    "@torch.no_grad()\n",
    "def predict_bilstm(text):\n",
    "    inputs = encode_text(text, muril_tokenizer)\n",
    "    start = time.time()\n",
    "    logits = bilstm_model(inputs[\"input_ids\"], inputs[\"attention_mask\"])\n",
    "    elapsed = (time.time()-start)*1000\n",
    "    probs = F.softmax(logits, dim=1).cpu().numpy()[0]\n",
    "    return probs, elapsed\n",
    "\n",
    "# =========================\n",
    "# CHARTS\n",
    "# =========================\n",
    "\n",
    "def make_prob_chart(prob_dict, title):\n",
    "    fig, ax = plt.subplots(figsize=(6,4))\n",
    "    ax.bar(list(prob_dict.keys()), list(prob_dict.values()))\n",
    "    ax.set_ylim(0,1)\n",
    "    ax.set_title(title)\n",
    "    plt.tight_layout()\n",
    "    return fig\n",
    "\n",
    "\n",
    "def make_comparison_chart(t,d,b):\n",
    "    labels=[\"Negative\",\"Neutral\",\"Positive\"]\n",
    "    x=list(range(3))\n",
    "    width=0.25\n",
    "    fig,ax=plt.subplots(figsize=(8,4))\n",
    "    ax.bar([i-width for i in x],t,width,label=\"Teacher\")\n",
    "    ax.bar(x,d,width,label=\"DistilBERT\")\n",
    "    ax.bar([i+width for i in x],b,width,label=\"BiLSTM\")\n",
    "    ax.set_xticks(x)\n",
    "    ax.set_xticklabels(labels)\n",
    "    ax.set_ylim(0,1)\n",
    "    ax.legend()\n",
    "    plt.tight_layout()\n",
    "    return fig\n",
    "\n",
    "# =========================\n",
    "# RESULT FORMAT\n",
    "# =========================\n",
    "\n",
    "def format_single_result(model_name, probs, elapsed, text, model_fn):\n",
    "\n",
    "    pred_idx = int(probs.argmax())\n",
    "    pred_label = LABEL_MAP[pred_idx]\n",
    "    confidence = float(probs[pred_idx])\n",
    "\n",
    "    final_label,_ = trust_decision(pred_label, confidence)\n",
    "    failure_msg = detect_failure(text, confidence)\n",
    "\n",
    "    explanation = format_explanation(get_token_importance(text, model_fn))\n",
    "\n",
    "    if confidence > 0.75:\n",
    "        color = \"🟢\"\n",
    "    elif confidence > 0.5:\n",
    "        color = \"🟡\"\n",
    "    else:\n",
    "        color = \"🔴\"\n",
    "\n",
    "    return (\n",
    "        f\"**Model:** {model_name}\\n\\n\"\n",
    "        f\"**Prediction:** {final_label}\\n\\n\"\n",
    "        f\"**Confidence:** {color} {confidence:.4f}\\n\\n\"\n",
    "        f\"**Latency:** {elapsed:.2f} ms\\n\\n\"\n",
    "        f\"**Status:** {failure_msg}\\n\\n\"\n",
    "        f\"{explanation}\"\n",
    "    )\n",
    "\n",
    "# =========================\n",
    "# UI FUNCTIONS\n",
    "# =========================\n",
    "\n",
    "def run_selected_model(text, model_name, language):\n",
    "\n",
    "    if model_name==\"Teacher (MuRIL + LoRA)\":\n",
    "        probs,elapsed=predict_teacher(text)\n",
    "        fn=predict_teacher\n",
    "\n",
    "    elif model_name==\"Student (DistilBERT)\":\n",
    "        probs,elapsed=predict_distilbert(text)\n",
    "        fn=predict_distilbert\n",
    "\n",
    "    else:\n",
    "        probs,elapsed=predict_bilstm(text)\n",
    "        fn=predict_bilstm\n",
    "\n",
    "    result=format_single_result(model_name,probs,elapsed,text,fn)\n",
    "    fig=make_prob_chart(probs_to_dict(probs),f\"{model_name} Probabilities\")\n",
    "\n",
    "    return result,fig\n",
    "\n",
    "\n",
    "def compare_all_models(text,language):\n",
    "\n",
    "    tp,t1=predict_teacher(text)\n",
    "    dp,t2=predict_distilbert(text)\n",
    "    bp,t3=predict_bilstm(text)\n",
    "\n",
    "    rows=[]\n",
    "\n",
    "    for name,p,t in [\n",
    "        (\"Teacher (MuRIL + LoRA)\",tp,t1),\n",
    "        (\"Student (DistilBERT)\",dp,t2),\n",
    "        (\"Student (BiLSTM)\",bp,t3)\n",
    "    ]:\n",
    "\n",
    "        pred_idx=int(p.argmax())\n",
    "        confidence=float(p[pred_idx])\n",
    "\n",
    "        final_label,_ = trust_decision(LABEL_MAP[pred_idx], confidence)\n",
    "\n",
    "        rows.append({\n",
    "            \"Model\":name,\n",
    "            \"Prediction\":final_label,\n",
    "            \"Confidence\":round(confidence,4),\n",
    "            \"Negative\":round(float(p[0]),4),\n",
    "            \"Neutral\":round(float(p[1]),4),\n",
    "            \"Positive\":round(float(p[2]),4),\n",
    "            \"Latency (ms)\":round(float(t),2)\n",
    "        })\n",
    "\n",
    "    df=pd.DataFrame(rows)\n",
    "    fig=make_comparison_chart(tp,dp,bp)\n",
    "\n",
    "    return df,fig\n",
    "\n",
    "# =========================\n",
    "# EXAMPLES\n",
    "# =========================\n",
    "\n",
    "examples=[\n",
    "[\"ఈ సినిమా చాలా బాగుంది\",\"Teacher (MuRIL + LoRA)\",\"Telugu\"],\n",
    "[\"ಈ ಸಿನಿಮಾ ಚೆನ್ನಾಗಿದೆ\",\"Student (DistilBERT)\",\"Kannada\"],\n",
    "[\"हा चित्रपट खूप छान आहे\",\"Student (BiLSTM)\",\"Marathi\"]\n",
    "]\n",
    "\n",
    "# =========================\n",
    "# GRADIO UI\n",
    "# =========================\n",
    "\n",
    "with gr.Blocks(title=\"Trustworthy Multilingual Sentiment AI\") as demo:\n",
    "\n",
    "    gr.Markdown(\"# 🔐 Trustworthy Multilingual Sentiment AI\")\n",
    "\n",
    "    gr.Markdown(\"Includes: Confidence-based trust, explanations, and failure detection\")\n",
    "\n",
    "    with gr.Row():\n",
    "\n",
    "        with gr.Column():\n",
    "\n",
    "            text_input=gr.Textbox(label=\"Enter text\",lines=4)\n",
    "\n",
    "            model_choice=gr.Dropdown(\n",
    "                [\"Teacher (MuRIL + LoRA)\",\n",
    "                 \"Student (DistilBERT)\",\n",
    "                 \"Student (BiLSTM)\"],\n",
    "                value=\"Teacher (MuRIL + LoRA)\"\n",
    "            )\n",
    "\n",
    "            language_choice=gr.Dropdown(\n",
    "                [\"Telugu\",\"Kannada\",\"Marathi\"],\n",
    "                value=\"Telugu\"\n",
    "            )\n",
    "\n",
    "            predict_btn=gr.Button(\"Predict\")\n",
    "            compare_btn=gr.Button(\"Compare All Models\")\n",
    "\n",
    "        with gr.Column():\n",
    "\n",
    "            result_output=gr.Markdown()\n",
    "            chart_output=gr.Plot()\n",
    "\n",
    "    comparison_table=gr.Dataframe()\n",
    "    comparison_chart=gr.Plot()\n",
    "\n",
    "    gr.Examples(\n",
    "        examples=examples,\n",
    "        inputs=[text_input,model_choice,language_choice]\n",
    "    )\n",
    "\n",
    "    predict_btn.click(\n",
    "        fn=run_selected_model,\n",
    "        inputs=[text_input,model_choice,language_choice],\n",
    "        outputs=[result_output,chart_output]\n",
    "    )\n",
    "\n",
    "    compare_btn.click(\n",
    "        fn=compare_all_models,\n",
    "        inputs=[text_input,language_choice],\n",
    "        outputs=[comparison_table,comparison_chart]\n",
    "    )\n",
    "\n",
    "demo.launch()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [
    {
     "databundleVersionId": 15135346,
     "datasetId": 9147566,
     "sourceId": 14328509,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 31260,
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
