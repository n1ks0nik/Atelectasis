import torch
import torch.nn as nn
import torch.optim as optim
from B_training_teacher_classifier import DeiTClassifierWSOL, train_teacher


# --- 7. Fine-tuning ---
if __name__ == "__main__":
    import os
    from A_making_data_loader_for_classification import classification_train_loader, classification_val_loader

    # 1. Model and Device
    teacher_model = DeiTClassifierWSOL(num_classes=3, pretrained=True, scm_blocks=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher_model = teacher_model.to(device)

    # 2. Optionally load checkpoint ONCE
    checkpoint_path = r"C:\Users\CYBER ARTEL\PycharmProjects\atelectasis_classification_and_detection\best_deit_scm_model — копия.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        teacher_model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found, starting from scratch.")

    # 3. Optimizer, Loss, Scheduler
    scm_params = []
    backbone_params = []
    for name, param in teacher_model.named_parameters():
        if 'scm' in name:
            scm_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_teacher = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 2e-6},
        {'params': scm_params, 'lr': 2e-5
         }  # 10x higher for SCM
    ], weight_decay=5e-4)
    weights = torch.tensor([0.7045, 0.1181, 0.1774], device=device)
    criterion_teacher = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer_teacher, step_size=10, gamma=0.1)
    num_epochs = 14

    # 4. Training
    train_teacher(
        model=teacher_model,
        train_loader=classification_train_loader,
        val_loader=classification_val_loader,
        num_epochs=num_epochs,
        optimizer=optimizer_teacher,
        criterion=criterion_teacher,
        scheduler=scheduler,
        device=device
    )