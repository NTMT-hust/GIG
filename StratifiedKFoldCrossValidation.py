from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    average_precision_score
)
from additional_function import *
from ImbalancedImageDataset import ImbalancedImageDataset
from EfficientNetB1Classifier import EfficientNetB1Classifier
from FocalLoss import FocalLoss
from BetaCalibrator import BetaCalibrator
from Resnet50 import ResNetClassifier
import gc
from GiG import *
from pathlib import Path
from AwareAugmentation import aware_augmentation
import pandas as pd
from tqdm import tqdm
from GiG import *

class StratifiedKFoldCrossValidation:
    def __init__(self,     
                model_name,
                dataset_path,
                k_folds=5,
                num_epochs=20,
                freeze_epochs=5,
                batch_size=32,
                lr=0.0001,
                weight_decay=1e-4,
                dropout_rate=0.3,
                focal_gamma=2.0,
                use_class_aware_aug=True,
                use_weighted_sampling=True,
                use_beta_calibration=True,
                calculate_cluster_metrics_flag=False,
                random_seed=42,
                lambda1 = 0.005):
        
        # Store parameters
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.k_folds = k_folds
        self.num_epochs = num_epochs
        self.freeze_epochs = freeze_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.focal_gamma = focal_gamma
        self.use_class_aware_aug = use_class_aware_aug
        self.use_weighted_sampling = use_weighted_sampling
        self.use_beta_calibration = use_beta_calibration
        self.calculate_cluster_metrics_flag = calculate_cluster_metrics_flag
        self.lambda1 = lambda1
        self.random_seed = random_seed
        
        # Set seeds
        self._set_seed(random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmup_epochs = 3
        self.mask = None
        self.cal_heatmap = False


    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        # 1. Load Dataset
        image_paths, labels, class_names, num_classes = load_dataset_from_folder(self.dataset_path)
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        class_counts = np.bincount(labels)
        # Set random seeds
        random_seed = self.random_seed
        use_class_aware_aug = self.use_class_aware_aug
        use_weighted_sampling = self.use_weighted_sampling
        calculate_cluster_metrics_flag = self.calculate_cluster_metrics_flag
        use_beta_calibration = self.use_beta_calibration
        
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        print(f'\n{"="*60}')
        print(f'Using device: {device}')
        print(f'Random seed: {random_seed}')
        print(f'Class-aware augmentation: {use_class_aware_aug}')
        print(f'Weighted sampling: {use_weighted_sampling}')
        print(f'Beta Calibration: {use_beta_calibration}')
        print(f'Calculate cluster metrics: {calculate_cluster_metrics_flag}')
        print(f'{"="*60}')

        # 2. Setup Transforms
        transform = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 3. Tách tập Test 20% trước (Stratified)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_seed)
        trainval_idx, test_idx = next(sss.split(image_paths, labels))

        test_image_paths = image_paths[test_idx]
        test_labels_arr  = labels[test_idx]
        image_paths      = image_paths[trainval_idx]
        labels           = labels[trainval_idx]
        class_counts     = np.bincount(labels)

        print(f'\n{"="*60}')
        print(f'Total samples   : {len(trainval_idx) + len(test_idx)}')
        print(f'Train+Val (80%) : {len(trainval_idx)}')
        print(f'Test      (20%) : {len(test_idx)}')
        print(f'{"="*60}')

        # K-Fold Initialization (trên 80% còn lại)
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_seed)
        all_fold_models = []
        all_val_indices = []
        all_calibrators = {} # Changed to dict to map fold_idx -> scaler
        fold_results = []
        # gene_dfs = [
        #     (pd.read_csv("C:\\Users\\Admin\\Downloads\\ResultTest_Aligned\\gene_coordinates_CNV.csv"), "CNV"),
        #     (pd.read_csv("C:\\Users\\Admin\\Downloads\\ResultTest_Aligned\\gene_coordinates_Methylation.csv"), "Methylation"),
        #     (pd.read_csv("C:\\Users\\Admin\\Downloads\\ResultTest_Aligned\\gene_coordinates_mRNA.csv"), "mRNA"),
        # ]


        # 4. Chuẩn bị Test DataLoader (dùng chung trong mọi fold)
        test_class_counts = np.bincount(labels)
        ds_test = ImbalancedImageDataset(
            test_image_paths, test_labels_arr, test_class_counts, transform=transform
        )
        test_loader = DataLoader(ds_test, batch_size=self.batch_size, shuffle=False)

        # Dict lưu kết quả test per-fold (trước & sau calibration)
        fold_test_results = {}

        # 5. Fold Loop
        for fold, (train_ids, val_ids) in enumerate(skf.split(image_paths, labels)):
            print(f'\n{"="*20} FOLD {fold + 1}/{self.k_folds} {"="*20}')
            all_val_indices.append(val_ids)
            
            # Setup loaders
            train_labels = labels[train_ids]
            train_class_counts = np.bincount(train_labels, minlength=num_classes)
            
            ds_train = ImbalancedImageDataset(image_paths[train_ids], train_labels, train_class_counts, 
                                            transform=transform)
            ds_val = ImbalancedImageDataset(image_paths[val_ids], labels[val_ids], train_class_counts, 
                                            transform=transform)

            train_labels = labels[train_ids]
            class_pools = {c: [] for c in range(num_classes)}
            for local_idx, label in enumerate(train_labels):
                class_pools[label].append(local_idx)

            if self.use_weighted_sampling:
                sampler = WeightedRandomSampler(get_sample_weights(train_labels, train_class_counts), 
                                                len(train_ids), replacement=True)
                train_loader = DataLoader(ds_train, batch_size=self.batch_size, sampler=sampler)
            else:
                train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
            
            val_loader = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False)

            # Model & Optimization
            if self.model_name == "EfficientNetB1Classifier":
                model = EfficientNetB1Classifier(num_classes=num_classes, dropout_rate=self.dropout_rate).to(self.device)
            elif self.model_name == "Resnet50":
                model = ResNetClassifier(num_classes=num_classes, dropout_rate=self.dropout_rate).to(self.device)
            model = nn.DataParallel(model)
            model.to(self.device)
            class_weights = calculate_class_weights(train_labels, num_classes).to(self.device)
            criterion = FocalLoss(num_classes, alpha= class_weights, gamma=self.focal_gamma)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr= self.lr,
                momentum=0.9,
                weight_decay= self.weight_decay,
                nesterov=True
            )

            # ---- Warmup + Cosine Scheduler ----
            if self.warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.05,   # bắt đầu từ 10% lr
                    total_iters=self.warmup_epochs
                )

                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.num_epochs - self.warmup_epochs
                )

                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_epochs]
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.num_epochs
                )

            # Training
            best_val_f1 = -1
            fold_history = defaultdict(list)

            for epoch in range(self.num_epochs):
                
                print(f'\nEpoch {epoch+1}/{self.num_epochs}', end='')

                if epoch == self.freeze_epochs and self.freeze_epochs > 0:
                    print(' [Unfreezing backbone]')
                    model.module.unfreeze_backbone()
                    optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=0.5, patience=3)
                    # optimizer = torch.optim.SGD(
                    #     model.parameters(),
                    #     lr= self.lr,
                    #     momentum=0.9,
                    #     weight_decay= self.weight_decay,
                    #     nesterov=True
                    # )
                    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= self.num_epochs)
                else:
                    print()

                # Train
                if (epoch % 3 == 0 and epoch >=9):
                    self.cal_heatmap = True
                train_loss, train_labels_epoch, train_preds, train_probs = train_epoch(model, train_loader, criterion, optimizer, device, class_pools, ds_train, use_class_aware_aug, self.cal_heatmap, self.mask, self.lambda1)
                train_metrics = calculate_comprehensive_metrics(train_labels_epoch, train_preds, train_probs, num_classes, class_names)
                # 2 Generate GradCAM from validation set
                global_heatmap = generate_epoch_gradcam(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    epoch=epoch,
                    class_names=class_names,
                    output_path=Path("GradCAM")
                )

                # 3 Update mask (after warmup)
                if epoch > 3 and global_heatmap is not None:
                    self.mask = build_mask_from_heatmap(global_heatmap, device)
                    print("Updated explanation mask.")

                # Validate
                val_loss, val_labels_epoch, val_preds, val_probs = validate_epoch(model, val_loader, criterion, device, fold)
                val_metrics = calculate_comprehensive_metrics(val_labels_epoch, val_preds, val_probs, num_classes, class_names)
                
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                # Save history
                fold_history['train_loss'].append(train_loss)
                fold_history['train_acc'].append(train_metrics['accuracy'])
                fold_history['train_auc'].append(train_metrics.get('roc_auc', train_metrics.get('roc_auc_ovr', 0)))
                fold_history['train_sens'].append(train_metrics['sensitivity'])
                fold_history['train_spec'].append(train_metrics['specificity'])
                fold_history['train_f1'].append(train_metrics['f1_macro'])

                fold_history['val_loss'].append(val_loss)
                fold_history['val_acc'].append(val_metrics['accuracy'])
                fold_history['val_auc'].append(val_metrics.get('roc_auc', val_metrics.get('roc_auc_ovr', 0)))
                fold_history['val_sens'].append(val_metrics['sensitivity'])
                fold_history['val_spec'].append(val_metrics['specificity'])
                fold_history['val_f1'].append(val_metrics['f1_macro'])
                print(f'Train - Loss: {train_loss:.4f} | Acc: {train_metrics["accuracy"]:.2f}% | '
                    f'AUC: {train_metrics.get("roc_auc", train_metrics.get("roc_auc_ovr", 0)):.4f} | '
                    f'F1: {train_metrics["f1_macro"]:.2f}%')
                print(f'Val   - Loss: {val_loss:.4f} | Acc: {val_metrics["accuracy"]:.2f}% | '
                    f'AUC: {val_metrics.get("roc_auc", val_metrics.get("roc_auc_ovr", 0)):.4f} | '
                    f'F1: {val_metrics["f1_macro"]:.2f}%')
                
                v_metrics = calculate_comprehensive_metrics(val_labels_epoch, val_preds, val_probs, num_classes, class_names)
                
                curr_f1 = v_metrics.get('f1_macro', v_metrics.get('f1', 0))

                if curr_f1 > best_val_f1:
                    best_val_f1 = curr_f1
                    best_metrics = v_metrics
                    torch.save(model.state_dict(),f'best_model_fold{fold}.pth')

            # Calibration
            print(f"Fold {fold} training complete. Generating Visualizations...")
            model.load_state_dict(torch.load(f'best_model_fold{fold}.pth', map_location=self.device))
            model.to(self.device)   # 🔑 REQUIRED
            model.eval()

            # ── Đánh giá Test TRƯỚC calibration (raw best model) ──────────────
            print(f'\n  [Fold {fold+1}] Evaluating best model on TEST SET (before calibration)...')
            raw_test_probs_list, raw_test_labels_list = [], []
            with torch.no_grad():
                for t_inputs, t_labels in tqdm(test_loader, desc='  Test (raw)', leave=False):
                    t_inputs = t_inputs.to(self.device)
                    t_logits = model(t_inputs)
                    t_probs  = torch.softmax(t_logits, dim=1)
                    raw_test_probs_list.append(t_probs.cpu().numpy())
                    raw_test_labels_list.extend(t_labels.numpy())

            raw_test_probs  = np.concatenate(raw_test_probs_list, axis=0)
            raw_test_labels = np.array(raw_test_labels_list)
            raw_test_preds  = np.argmax(raw_test_probs, axis=1)

            test_metrics_before = calculate_comprehensive_metrics(
                raw_test_labels, raw_test_preds, raw_test_probs, num_classes, class_names
            )
            print(f'  [Fold {fold+1}] TEST (before calib) — '
                  f'Acc: {test_metrics_before["accuracy"]:.2f}% | '
                  f'AUC: {test_metrics_before.get("roc_auc", test_metrics_before.get("roc_auc_ovr", 0)):.4f} | '
                  f'F1:  {test_metrics_before["f1_macro"]:.2f}%')
            plot_confusion_matrix(
                test_metrics_before['confusion_matrix'],
                class_names,
                title=f'Fold {fold+1} — Test (Before Calibration)',
                save_path=f'confusion_matrix_fold{fold+1}_test_before_calib.png'
            )

            # gig_out = Path(f"GIG_CSV/Fold_{fold}")
            # gig_out.mkdir(parents=True, exist_ok=True)

            # sample_counter = 0

            # for batch_inputs, batch_labels in val_loader:
            #     batch_inputs = batch_inputs.to(self.device)
            #     batch_labels = batch_labels.to(self.device)

            #     for i in range(batch_inputs.size(0)):
            #         x = batch_inputs[i:i+1]
            #         y_true = batch_labels[i].item()

            #         with torch.no_grad():
            #             logits = model(x)
            #             y_pred = logits.argmax(dim=1).item()

            #         # ⚠️ Bật gradient cho GIG
            #         x.requires_grad_(True)

            #         attr = compute_gig(model, x, y_pred)

            #         csv_path = gig_out / f"sample_{sample_counter}.csv"

            #         save_gene_csv(
            #             attr=attr.cpu(),
            #             gene_dfs=gene_dfs,
            #             output_csv=csv_path,
            #             fold=fold,
            #             sample_id=sample_counter,
            #             pred_label=y_pred,
            #             true_label=y_true
            #         )

            #         sample_counter += 1

            if use_beta_calibration:
                print(f'\n  [Fold {fold+1}] Fitting Beta Calibrator on validation set...')
                scaler = BetaCalibrator(num_classes=num_classes)
                scaler.fit_loader(model, val_loader, self.device)
                all_calibrators[fold] = scaler

                # ── Lấy calibrated probs trên val set ─────────────────────────
                print(f'  [Fold {fold+1}] Collecting calibrated probabilities on validation set...')
                calib_probs_list, calib_labels_list = [], []
                model.eval()
                with torch.no_grad():
                    for cal_inputs, cal_labels in val_loader:
                        cal_inputs = cal_inputs.to(self.device)
                        cal_logits = model(cal_inputs)
                        cal_probs  = scaler.calibrate_probs(cal_logits)   # Tensor [B, C]
                        calib_probs_list.append(cal_probs.cpu().numpy())
                        calib_labels_list.extend(cal_labels.numpy())

                calib_probs_all  = np.concatenate(calib_probs_list, axis=0)  # [N, C]
                calib_labels_all = np.array(calib_labels_list)                # [N]

                # ── Tune thresholds per class trên val ────────────────────────
                scaler.tune_thresholds(calib_probs_all, calib_labels_all)

                # ── Dự đoán với thresholds đã tune ────────────────────────────
                calib_preds_all = scaler.predict_with_thresholds(calib_probs_all)

                calib_metrics = calculate_comprehensive_metrics(
                    calib_labels_all, calib_preds_all, calib_probs_all, num_classes, class_names
                )
                print(f'  [Fold {fold+1}] Calibrated Val (with tuned thresholds) — '
                      f'Acc: {calib_metrics["accuracy"]:.2f}% | '
                      f'AUC: {calib_metrics.get("roc_auc", calib_metrics.get("roc_auc_ovr", 0)):.4f} | '
                      f'F1: {calib_metrics["f1_macro"]:.2f}%')
                # Dùng calibrated metrics làm best_metrics của fold
                best_metrics = calib_metrics

                # ── Đánh giá Test SAU calibration ─────────────────────────────
                print(f'  [Fold {fold+1}] Evaluating Beta-calibrated model on TEST SET (after calibration)...')
                cal_test_probs_list = []
                with torch.no_grad():
                    for t_inputs, _ in tqdm(test_loader, desc='  Test (calib)', leave=False):
                        t_inputs  = t_inputs.to(self.device)
                        t_logits  = model(t_inputs)
                        t_probs   = scaler.calibrate_probs(t_logits)
                        cal_test_probs_list.append(t_probs.cpu().numpy())

                cal_test_probs = np.concatenate(cal_test_probs_list, axis=0)
                cal_test_preds = scaler.predict_with_thresholds(cal_test_probs)  # ← tuned thresholds

                test_metrics_after = calculate_comprehensive_metrics(
                    raw_test_labels, cal_test_preds, cal_test_probs, num_classes, class_names
                )
                print(f'  [Fold {fold+1}] TEST (after  calib) — '
                      f'Acc: {test_metrics_after["accuracy"]:.2f}% | '
                      f'AUC: {test_metrics_after.get("roc_auc", test_metrics_after.get("roc_auc_ovr", 0)):.4f} | '
                      f'F1:  {test_metrics_after["f1_macro"]:.2f}%')
                print(f'  [Fold {fold+1}] Thresholds used: {np.round(scaler.thresholds, 3).tolist()}')
                plot_confusion_matrix(
                    test_metrics_after['confusion_matrix'],
                    class_names,
                    title=f'Fold {fold+1} — Test (After Beta Calibration + Threshold Tuning)',
                    save_path=f'confusion_matrix_fold{fold+1}_test_after_calib.png'
                )

                fold_test_results[fold] = {
                    'before_calibration': test_metrics_before,
                    'after_calibration' : test_metrics_after,
                    'thresholds'        : scaler.thresholds.tolist(),
                }

            else:
                # Không dùng beta calibration → chỉ lưu before
                fold_test_results[fold] = {
                    'before_calibration': test_metrics_before,
                    'after_calibration' : None,
                    'thresholds'        : None,
                }

            all_fold_models.append(model)
            fold_results.append({
                'fold'        : fold + 1,
                'metrics'     : best_metrics,
                'test_metrics': fold_test_results[fold],
                'history'     : dict(fold_history),
            })
            print_detailed_metrics(best_metrics, class_names, fold+1)
            
            # Plot confusion matrix for this fold
            plot_confusion_matrix(
                best_metrics['confusion_matrix'],
                class_names,
                title=f'Fold {fold+1} Confusion Matrix',
                save_path=f'confusion_matrix_fold_{fold+1}.png'
            )
            
            # Delete the optimizer objects
            del optimizer

            # Force garbage collection and empty CUDA cache
            gc.collect()
            torch.cuda.empty_cache()

        # 6. Tổng kết kết quả Test per-fold & Ensemble
        print(f'\n{"="*60}')
        print(f'PER-FOLD TEST SUMMARY')
        print(f'{"="*60}')
        for f_idx, ftr in fold_test_results.items():
            bm = ftr['before_calibration']
            am = ftr['after_calibration']
            print(f'  Fold {f_idx+1}:')
            print(f'    Before calib — Acc: {bm["accuracy"]:.2f}%  AUC: {bm.get("roc_auc", bm.get("roc_auc_ovr",0)):.4f}  F1: {bm["f1_macro"]:.2f}%')
            if am is not None:
                t_str = str(np.round(ftr['thresholds'], 3).tolist()) if ftr['thresholds'] else 'N/A'
                print(f'    After  calib — Acc: {am["accuracy"]:.2f}%  AUC: {am.get("roc_auc", am.get("roc_auc_ovr",0)):.4f}  F1: {am["f1_macro"]:.2f}%  thresholds={t_str}')

        print(f'\n{"="*60}')
        print(f'ENSEMBLE ON HELD-OUT TEST SET (all folds)')
        print(f'{"="*60}')

        # Ensemble predictions từ tất cả các fold model (dùng calibrated probs nếu có)
        all_test_probs = []
        for f_idx, fold_model in enumerate(all_fold_models):
            fold_model.to(self.device)
            fold_model.eval()

            scaler = all_calibrators.get(f_idx)   # None nếu không dùng temperature scaling
            fold_probs = []

            with torch.no_grad():
                for inputs, _ in tqdm(test_loader, desc=f'Test Fold {f_idx+1}', leave=False):
                    inputs = inputs.to(self.device)
                    logits = fold_model(inputs)        # raw logits [B, C]
                    if scaler is not None:
                        # calibrate_probs: scale logits rồi softmax
                        probs = scaler.calibrate_probs(logits)
                    else:
                        probs = torch.softmax(logits, dim=1)
                    fold_probs.append(probs.cpu().numpy())

            fold_model.cpu()
            torch.cuda.empty_cache()
            all_test_probs.append(np.concatenate(fold_probs))

        # Average ensemble probabilities
        ensemble_test_probs = np.mean(all_test_probs, axis=0)

        # Ensemble threshold tuning: lấy thresholds trung bình từ tất cả folds (nếu có)
        all_thresh = [all_calibrators[f].thresholds for f in sorted(all_calibrators.keys())
                      if all_calibrators[f].thresholds is not None]
        if all_thresh:
            mean_thresholds = np.mean(all_thresh, axis=0)
            # Adjusted argmax với mean thresholds
            ensemble_test_preds = np.argmax(ensemble_test_probs - mean_thresholds[np.newaxis, :], axis=1)
            print(f'  [Ensemble] Mean thresholds across folds: {np.round(mean_thresholds, 3).tolist()}')
        else:
            ensemble_test_preds = np.argmax(ensemble_test_probs, axis=1)

        test_metrics = calculate_comprehensive_metrics(
            test_labels_arr, ensemble_test_preds, ensemble_test_probs, num_classes, class_names
        )

        print('\n===== TEST SET RESULTS (Ensemble of all folds) =====')
        print_detailed_metrics(test_metrics, class_names)
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names,
            title='Test Set Confusion Matrix',
            save_path='confusion_matrix_test.png'
        )

        return fold_results, all_fold_models, test_metrics, fold_test_results, class_names, all_calibrators

def train_epoch(model, dataloader, criterion, optimizer, device, class_pools, ds_train, use_class_aware_aug, cal_heatmap, mask, lambda1):
    model.train()
    model.to(device)
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        if use_class_aware_aug:
            augmented_images = []
            for i in range(inputs.size(0)):
                img, lbl = aware_augmentation(
                    inputs[i],
                    labels[i].item(),
                    class_pools,
                    ds_train
                )
                augmented_images.append(img)
            inputs = torch.stack(augmented_images)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad_(True)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Standard cross entropy
        labels = labels.to(device).long()
        ce_loss = criterion(outputs, labels)
        # Apply mask (Right Reasons)
        if mask is not None and  cal_heatmap:

            probs = torch.softmax(outputs, dim=1)
            S = torch.log(probs + 1e-8).sum(dim=1)

            grads = torch.autograd.grad(
                outputs=S.sum(),
                inputs=inputs,
                create_graph=True
            )[0]

            # expand mask to batch size + channels
            batch_mask = mask.repeat(inputs.size(0), inputs.size(1), 1, 1)

            grad_penalty = ((batch_mask * grads) ** 2).mean()

            loss = ce_loss + (lambda1 * grad_penalty)

        else:
            loss = ce_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return epoch_loss, all_labels, all_preds, all_probs

def validate_epoch(model, dataloader, criterion, device, fold_idx):
    model.to(device)
    model.eval()

    backbone = model.module.backbone

    if hasattr(backbone, "blocks"):
        target_block = backbone.blocks[-1]
    elif hasattr(backbone, "layer4"):
        target_block = backbone.layer4
    else:
        raise ValueError("Unknown backbone type")

    for p in target_block.parameters():
        p.requires_grad = True

    output_path = Path(f"Explanations/Fold_{fold_idx}")
    output_path.mkdir(parents=True, exist_ok=True)

    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    all_outputs = []  # giữ tensor

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation', leave=False):

            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)

            # ✅ Giữ tensor
            all_outputs.append(outputs.detach())
            all_labels.append(labels.detach())

            # metrics dùng numpy thì convert riêng
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())

    all_output_s = torch.cat(all_outputs, dim=0)
    all_label_s = torch.cat(all_labels, dim=0)

    return (
        running_loss / len(dataloader),
        all_label_s.cpu().numpy(),
        np.array(all_preds),
        np.array(all_probs),
    )

def generate_fold_gradcam(model, dataloader, device, fold_idx, class_names, save_limit=20):
    model.to(device)
    model.eval()
    
    # 1. Target the last convolutional layer
    # For timm EfficientNet, 'conv_head' is the final feature map layer
    backbone = model.module.backbone
    if hasattr(backbone, "conv_head"):            # EfficientNet
        target_layer = backbone.conv_head
    elif hasattr(backbone, "blocks"):             # ConvNeXt / some timm models
        target_layer = backbone.blocks[-1]
    elif hasattr(backbone, "layer4"):              # ResNet
        target_layer = backbone.layer4[-1]
    else:
        raise ValueError("Unsupported backbone for Grad-CAM")

    
    # 2. Initialize GradCAM helper
    cam_extractor = GuidedIntegratedGradients(model, target_layer)
    
    output_path = Path(f"GradCAM_Results/Fold_{fold_idx}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"--> Generating Grad-CAM for Fold {fold_idx}...")
    all_heatmaps = []
    # Process samples
    for i in range(min(len(dataloader.dataset), save_limit)):
        img_tensor, label = dataloader.dataset[i]
        img_path = dataloader.dataset.image_paths[i]
        
        input_tensor = img_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True 

        # Generate heatmap using our helper
        with torch.enable_grad():
            heatmap, pred_idx = cam_extractor.generate(input_tensor, class_idx=None)

        # Save using the visualization function created earlier
        heatmap_resized = save_gradcam_image(
            img_path=img_path,
            heatmap=heatmap,
            pred_class=class_names[pred_idx],
            true_class=class_names[label],
            output_dir=output_path
        )
        if class_names[pred_idx] == class_names[label]:
            all_heatmaps.append((heatmap_resized, class_names[pred_idx]))

    # 3. CRITICAL: Remove hooks and free memory
    cam_extractor.remove_hooks()
    model.zero_grad()
    torch.cuda.empty_cache()
    
    return all_heatmaps

def save_gradcam_image(img_path, heatmap, pred_class, true_class, output_dir, alpha=0.4):
    """
    Overlays a Grad-CAM heatmap on the original image and saves it.
    
    Args:
        img_path: Path to the original image file.
        heatmap: 2D numpy array (float) from the Grad-CAM logic.
        pred_class: Predicted class name or index.
        true_class: Ground truth class name or index.
        output_dir: Folder to save the resulting image.
        alpha: Transparency of the heatmap (0.0 to 1.0).
    """
    # 1. Load original image
    raw_img = cv2.imread(str(img_path))
    if raw_img is None:
        print(f"Error: Could not load image {img_path}")
        return
    
    height, width, _ = raw_img.shape

    # 2. Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (width, height))

    # 3. Normalize and convert to 0-255 (uint8)
    # Ensure heatmap is in range [0, 1] before this
    heatmap_norm = np.uint8(255 * heatmap_resized)

    # 4. Apply JET colormap (Red = Hot/Important, Blue = Cold)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # 5. Overlay heatmap onto the original image
    # combined = original * (1 - alpha) + heatmap * alpha
    overlay = cv2.addWeighted(raw_img, 1 - alpha, heatmap_color, alpha, 0)

    # 6. Add Text labels (Optional but helpful for debugging)
    label_text = f"Pred: {pred_class} | True: {true_class}"
    cv2.putText(overlay, label_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 7. Save the result
    sample_name = Path(img_path).stem
    save_path = Path(output_dir) / f"{sample_name}_gradcam.png"
    cv2.imwrite(str(save_path), overlay)
    return heatmap_resized

def generate_epoch_gradcam(model, dataloader, device, epoch, class_names, output_path):
    model.to(device)
    model.eval()
    
    # 1. Target the last convolutional layer
    # For timm EfficientNet, 'conv_head' is the final feature map layer
    backbone = model.module.backbone
    if hasattr(backbone, "conv_head"):            # EfficientNet
        target_layer = backbone.conv_head
    elif hasattr(backbone, "blocks"):             # ConvNeXt / some timm models
        target_layer = backbone.blocks[-1]
    elif hasattr(backbone, "layer4"):              # ResNet
        target_layer = backbone.layer4[-1]
    else:
        raise ValueError("Unsupported backbone for Grad-CAM")

    
    # 2. Initialize GradCAM helper
    cam_extractor = GuidedIntegratedGradients(model)
    

    output_path.mkdir(parents=True, exist_ok=True)

    pos_heatmaps = []
    neg_heatmaps = []
    # Process samples
    for i in range(min(len(dataloader.dataset))):
        img_tensor, label = dataloader.dataset[i]
        img_path = dataloader.dataset.image_paths[i]
        
        input_tensor = img_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True 

        # Generate heatmap using our helper
        with torch.enable_grad():
            heatmap, pred_idx, raw_heatmap = cam_extractor.generate(input_tensor, class_idx=None)

        if class_names[pred_idx] == class_names[label]:
            pos_heatmaps.append((raw_heatmap, class_names[pred_idx]))
        else:
            neg_heatmaps.append((raw_heatmap, class_names[label]))
        del heatmap, raw_heatmap, input_tensor
        torch.cuda.empty_cache()

    # 3. CRITICAL: Remove hooks and free memory
    cam_extractor.remove_hooks()
    model.zero_grad()
    torch.cuda.empty_cache()
        
    # keep only heatmaps
    pos_heatmaps = [h for (h, _) in pos_heatmaps]
    neg_heatmaps = [h for (h, _) in neg_heatmaps]

    fallback_shape = None
    if len(pos_heatmaps) > 0:
        fallback_shape = pos_heatmaps[0].shape
    elif len(neg_heatmaps) > 0:
        fallback_shape = neg_heatmaps[0].shape

    pos_heatmap = safe_mean_stack(pos_heatmaps, fallback_shape)
    neg_heatmap = safe_mean_stack(neg_heatmaps, fallback_shape)

    if pos_heatmap is None:
        return None

    bin_pos_heatmap = (pos_heatmap >= 0).astype(np.float32)
    bin_reverse_pos_heatmap = (pos_heatmap <= 0).astype(np.float32)
    bin_reverse_pos_heatmap = -bin_reverse_pos_heatmap
    bin_neg_heatmap = (neg_heatmap <= 0).astype(np.float32)

    global_heatmap = bin_neg_heatmap - bin_pos_heatmap + bin_reverse_pos_heatmap
    return global_heatmap


def build_mask_from_heatmap(global_heatmap, device):

    mask_np = global_heatmap

    # If 3D (C,H,W) → collapse channel
    if mask_np.ndim == 3:
        mask_np = mask_np.mean(axis=0)

    if mask_np.ndim != 2:
        raise ValueError(f"Unexpected heatmap shape: {mask_np.shape}")

    mask_tensor = torch.tensor(mask_np, dtype=torch.float32, device=device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    return mask_tensor

def safe_mean_stack(heatmaps, fallback_shape=None):
    if len(heatmaps) == 0:
        if fallback_shape is None:
            return None
        return np.zeros(fallback_shape, dtype=np.float32)
    return np.mean(np.stack(heatmaps), axis=0)
