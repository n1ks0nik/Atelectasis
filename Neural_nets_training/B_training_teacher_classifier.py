from torch.optim import optimizer, lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm import create_model
import os
import random
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                          train_precisions, val_precisions,
                          train_recalls, val_recalls,
                          train_f1s, val_f1s,
                          epoch=None, save_dir="metrics_plots"):
    os.makedirs(save_dir, exist_ok=True)

    def plot_metric(train_values, val_values, ylabel, name):
        plt.figure()
        plt.plot(train_values, label="Train")
        plt.plot(val_values, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} per Epoch")
        plt.legend()
        suffix = f"_epoch{epoch}.png" if epoch else "_final.png"
        plt.savefig(os.path.join(save_dir, f"{name}{suffix}"))
        plt.close()

    plot_metric(train_losses, val_losses, "Loss", "loss")
    plot_metric(train_accuracies, val_accuracies, "Accuracy", "accuracy")
    plot_metric(train_precisions, val_precisions, "Precision", "precision")
    plot_metric(train_recalls, val_recalls, "Recall", "recall")
    plot_metric(train_f1s, val_f1s, "F1 Score", "f1_score")
# ---- SCM block ----

class ActivationDiffusionBlock(nn.Module):
    def __init__(self, patch_h=14, patch_w=14):
        super().__init__()
        # Initialize with stronger values
        self.lmbd = nn.Parameter(torch.tensor(2.0))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.patch_h, self.patch_w = patch_h, patch_w
        self.N = patch_h * patch_w
        self.register_buffer("A", self._build_adjacency(patch_h, patch_w))
        self.register_buffer("I", torch.eye(self.N).unsqueeze(0))

    def _build_adjacency(self, H, W):
        A = torch.zeros(H * W, H * W)
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= ni < H and 0 <= nj < W:
                        A[idx, ni * W + nj] = 1
        return A.unsqueeze(0)

    def forward(self, S, attn):
        B, C, H, W = S.shape

        # Scale attention for greater effect
        attn = attn * 5.0

        S_flat = F.normalize(S.view(B, C, -1), dim=1)
        E = torch.matmul(S_flat.transpose(1, 2), S_flat)
        E = (E + 1) / 2

        A = self.A.expand(B, -1, -1)
        D = torch.diag_embed(A.sum(-1))
        L = (D - A) * (self.lmbd * E - 1)

        # Increase stability of the diffusion process
        L = L + 1e-6 * self.I
        X = L.transpose(-1, -2) * 0.01

        for _ in range(4):
            X_new = X @ (2 * self.I - L @ X)
            X = X_new

        attn_flat = attn.view(B, 1, -1)
        F_diffused = torch.bmm(X, attn_flat.transpose(1, 2)).transpose(1, 2)

        # Add scaling to ensure the diffusion has meaningful effect
        F_diff = self.beta * (F_diffused - torch.tanh(F_diffused / (self.beta + 1e-8)))

        # Use residual to ensure gradient flow
        S_new = S * (1.0 + F_diff.view(B, 1, H, W))

        return S_new, F_diff.view(B, 1, H, W)


class SCM(nn.Module):
    def __init__(self, channels, patch_hw=14, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            ActivationDiffusionBlock(patch_h=patch_hw, patch_w=patch_hw)
            for _ in range(n_blocks)
        ])
        print(f"Initialized SCM with {n_blocks} blocks")

    def forward(self, S, attn):
        for i, block in enumerate(self.blocks):
            S, attn = block(S, attn)
        return S, attn


def get_deit_attention_map(attn_probs, patch_hw):
    # Make sure attn_probs is not None
    if attn_probs is None:
        raise ValueError("Attention probs are None - hook not working correctly")

    cls2patch = attn_probs[:, :, 0, 1:]
    return cls2patch.mean(dim=1).reshape(-1, 1, patch_hw, patch_hw)


def get_deit_semantic_map(tokens, patch_hw):
    patch_tokens = tokens[:, 1:, :]
    B, N, C = patch_tokens.shape
    return patch_tokens.transpose(1, 2).contiguous().view(B, C, patch_hw, patch_hw)


class DeiTWithAttn(nn.Module):
    def __init__(self, variant='deit_tiny_patch16_224', pretrained=True):
        super().__init__()
        self.model = create_model(variant, pretrained=pretrained)
        self.model.reset_classifier(0)
        self._attn = None

        # Better hook implementation
        def hook_fn(module, input, output):
            self._attn = module.attn_drop.output if hasattr(module.attn_drop, 'output') else None

        # Try extracting attention directly from the model
        for i, blk in enumerate(self.model.blocks):
            if i == len(self.model.blocks) - 1:  # Last block
                # Patch the forward method of the attention module
                orig_forward = blk.attn.forward

                def new_forward(self_attn, x):
                    B, N, C = x.shape
                    qkv = self_attn.qkv(x).reshape(B, N, 3, self_attn.num_heads, C // self_attn.num_heads).permute(2, 0,
                                                                                                                   3, 1,
                                                                                                                   4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    attn = (q @ k.transpose(-2, -1)) * self_attn.scale
                    attn_probs = attn.softmax(dim=-1)
                    # Store the attention probs
                    self_attn.attn_probs = attn_probs

                    x = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
                    x = self_attn.proj(x)
                    x = self_attn.proj_drop(x)
                    return x

                # Apply the patched forward method
                blk.attn.forward = new_forward.__get__(blk.attn, type(blk.attn))

    def forward(self, x):
        tokens = self.model.forward_features(x)
        # Extract attention from the last block
        attn_probs = self.model.blocks[-1].attn.attn_probs
        return tokens, attn_probs


class DeiTClassifierWSOL(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, patch_hw=14, scm_blocks=4):
        super().__init__()
        self.patch_hw = patch_hw
        self.backbone = DeiTWithAttn('deit_tiny_patch16_224', pretrained=pretrained)
        self.num_features = self.backbone.model.embed_dim
        self.scm = SCM(self.num_features, patch_hw=patch_hw, n_blocks=scm_blocks)
        self.head = nn.Linear(self.num_features, num_classes)
        print(f"Initialized DeiTClassifierWSOL with {num_classes} classes, {scm_blocks} SCM blocks")

    def forward(self, x):
        tokens, attn_probs = self.backbone(x)
        sem_map = get_deit_semantic_map(tokens, self.patch_hw)
        attn_map = get_deit_attention_map(attn_probs, self.patch_hw)

        # CRITICAL: Only use SCM during training
        if self.training:
            S_out, _ = self.scm(sem_map, attn_map)
            pooled = S_out.mean(dim=[2, 3])
        else:
            # During inference, don't use SCM, just use the backbone features
            pooled = sem_map.mean(dim=[2, 3])

        return self.head(pooled)

    @torch.no_grad()
    def localize(self, x, target_class=None):
        tokens, attn_probs = self.backbone(x)
        sem_map = get_deit_semantic_map(tokens, self.patch_hw)
        attn_map = get_deit_attention_map(attn_probs, self.patch_hw)
        _, F_out = self.scm(sem_map, attn_map)
        heat = F_out.squeeze(1)
        minval = heat.flatten(1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        maxval = heat.flatten(1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        return (heat - minval) / (maxval - minval + 1e-6)


def simple_save_localizations(model, device, data_loader, epoch, folder="localization_samples", num_samples=8,
                              class_of_interest=None):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        indices = random.sample(range(len(images)), min(num_samples, len(images)))
        images = images[indices]
        labels = labels[indices]
        break

    with torch.no_grad():
        heatmaps = model.localize(images, target_class=class_of_interest)

    for i, (img, label, heat) in enumerate(zip(images, labels, heatmaps)):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        heat_t = F.interpolate(heat.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear')[0, 0]
        heat_np = heat_t.cpu().numpy()
        overlay = np.clip(
            img_np * 0.6 + np.stack([heat_np, np.zeros_like(heat_np), np.zeros_like(heat_np)], axis=-1) * 0.4, 0, 1)
        Image.fromarray((overlay * 255).astype(np.uint8)).save(
            f"{folder}/epoch{epoch}_sample{i}_label{label.item()}.png")


def train_teacher(model, train_loader, val_loader, num_epochs=20, optimizer=None, criterion=None, device='cuda',
                  scheduler=None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses, train_accuracies, val_accuracies, val_losses = [], [], [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    best_val_accuracy, patience, trigger_times = 0, 10, 0

    # Add gradient debugging
    for name, param in model.named_parameters():
        if 'lmbd' in name or 'beta' in name:
            print(f"Initial {name}: {param.item():.4f}")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, all_train_preds, all_train_labels = 0, [], []
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Debug: Check gradients for SCM parameters
            # if random.random() < 0.1:  # Only check 10% of the time
            #     with torch.no_grad():
            #         for name, param in model.named_parameters():
            #             if 'lmbd' in name or 'beta' in name:
            #                 if param.grad is None:
            #                     print(f"WARNING: {name} has no gradient!")
            #                 elif torch.all(param.grad == 0):
            #                     print(f"WARNING: {name} has zero gradient!")
            #                 else:
            #                     print(f"{name} grad: {param.grad.item():.6f}, param: {param.item():.4f}")

            optimizer.step()
            total_train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_train_labels, all_train_preds)
        precision = precision_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        recall = recall_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)

        train_losses.append(total_train_loss / len(train_loader))
        train_accuracies.append(acc)
        train_precisions.append(precision)
        train_recalls.append(recall)
        train_f1s.append(f1)
        print(f"Train Loss: {train_losses[-1]:.4f}, Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")

        # Print SCM parameter values
        #with torch.no_grad():
            #for name, param in model.named_parameters():
                #if 'lmbd' in name or 'beta' in name:
                    #print(f"SCM param {name}: {param.item():.4f}")

        simple_save_localizations(model, device, train_loader, epoch + 1, class_of_interest=0)

        model.eval()
        total_val_loss, all_val_preds, all_val_labels = 0, [], []
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

        val_losses.append(total_val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        print(f"Val Loss: {val_losses[-1]:.4f}, Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), 'best_deit_scm_model — копия.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

        if scheduler: scheduler.step()
        torch.cuda.empty_cache()

        plot_and_save_metrics(train_losses, val_losses,
                              train_accuracies, val_accuracies,
                              train_precisions, val_precisions,
                              train_recalls, val_recalls,
                              train_f1s, val_f1s,
                              epoch=epoch + 1)

        plot_and_save_metrics(train_losses, val_losses,
                              train_accuracies, val_accuracies,
                              train_precisions, val_precisions,
                              train_recalls, val_recalls,
                              train_f1s, val_f1s)

if __name__ == "__main__":
    from A_making_data_loader_for_classification import classification_train_loader, classification_val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix learning rate and increase SCM blocks
    model = DeiTClassifierWSOL(num_classes=3, pretrained=True, patch_hw=14, scm_blocks=4).to(device)

    # Correct learning rate: 5e-5 (not 10e-5)
    # Use different learning rates
    scm_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if 'scm' in name:
            scm_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5},
        {'params': scm_params, 'lr': 5e-4}  # 10x higher for SCM
    ], weight_decay=5e-4)

    # Use a gentler scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 0.5 instead of 0.1

    weights = torch.tensor([0.7045, 0.1181, 0.1774], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    train_teacher(
        model=model,
        train_loader=classification_train_loader,
        val_loader=classification_val_loader,
        num_epochs=20,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device='cuda'
    )