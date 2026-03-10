from StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ProcessHeatMapResult import *
from torch import nn
import os

if __name__ == '__main__':
    dataset_path = "H:/My Drive/bioinfor_training/COAD_Aligned-20260309T160020Z-3-001/COAD_Aligned/dataset"

    model = StratifiedKFoldCrossValidation(
        model_name="EfficientNetB1Classifier",
        dataset_path=dataset_path,
        k_folds=5,
        num_epochs=200,
        freeze_epochs= 0,
        batch_size= 16,
        lr=0.01,
        weight_decay=1e-3,
        dropout_rate=0.5,
        focal_gamma= 1,
        use_class_aware_aug= True,
        use_weighted_sampling= True,
        use_beta_calibration= True,
        calculate_cluster_metrics_flag=False,
        random_seed=42
    )

    # ── Unpack return (6 values) ──────────────────────────────────────────────
    fold_results, fold_models, ensemble_metrics, fold_test_results, class_names, calibrators = model.run()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f'/n{"="*60}')
    print('FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'Classes     : {class_names}')
    print(f'Num folds   : {len(fold_results)}')

    # ── Per-fold val metrics ──────────────────────────────────────────────────
    print(f'/n{"─"*60}')
    print('PER-FOLD VALIDATION METRICS (best calibrated)')
    print(f'{"─"*60}')
    val_accs, val_aucs, val_f1s = [], [], []
    for result in fold_results:
        fold_num = result['fold']
        m = result['metrics']
        acc = m['accuracy']
        auc = m.get('roc_auc', m.get('roc_auc_ovr', 0))
        f1  = m['f1_macro']
        val_accs.append(acc); val_aucs.append(auc); val_f1s.append(f1)
        print(f'  Fold {fold_num}: Acc={acc:.2f}%  AUC={auc:.4f}  F1={f1:.2f}%')
    print(f'  {"─"*40}')
    print(f'  Mean : Acc={sum(val_accs)/len(val_accs):.2f}%  '
          f'AUC={sum(val_aucs)/len(val_aucs):.4f}  '
          f'F1={sum(val_f1s)/len(val_f1s):.2f}%')

    # ── Per-fold test metrics (before / after calibration) ───────────────────
    print(f'/n{"─"*60}')
    print('PER-FOLD TEST METRICS')
    print(f'{"─"*60}')
    test_before_accs, test_before_aucs, test_before_f1s = [], [], []
    test_after_accs,  test_after_aucs,  test_after_f1s  = [], [], []

    for fold_idx, ftr in fold_test_results.items():
        bm = ftr['before_calibration']
        am = ftr['after_calibration']
        thresholds = ftr['thresholds']

        b_acc = bm['accuracy']
        b_auc = bm.get('roc_auc', bm.get('roc_auc_ovr', 0))
        b_f1  = bm['f1_macro']
        test_before_accs.append(b_acc); test_before_aucs.append(b_auc); test_before_f1s.append(b_f1)

        print(f'  Fold {fold_idx+1}:')
        print(f'    Before calib — Acc={b_acc:.2f}%  AUC={b_auc:.4f}  F1={b_f1:.2f}%')
        if am is not None:
            a_acc = am['accuracy']
            a_auc = am.get('roc_auc', am.get('roc_auc_ovr', 0))
            a_f1  = am['f1_macro']
            test_after_accs.append(a_acc); test_after_aucs.append(a_auc); test_after_f1s.append(a_f1)
            print(f'    After  calib — Acc={a_acc:.2f}%  AUC={a_auc:.4f}  F1={a_f1:.2f}%')
            if thresholds:
                import numpy as np
                print(f'    Thresholds    — {[round(t, 3) for t in thresholds]}')

    print(f'  {"─"*40}')
    print(f'  Mean before: Acc={sum(test_before_accs)/len(test_before_accs):.2f}%  '
          f'AUC={sum(test_before_aucs)/len(test_before_aucs):.4f}  '
          f'F1={sum(test_before_f1s)/len(test_before_f1s):.2f}%')
    if test_after_accs:
        print(f'  Mean after : Acc={sum(test_after_accs)/len(test_after_accs):.2f}%  '
              f'AUC={sum(test_after_aucs)/len(test_after_aucs):.4f}  '
              f'F1={sum(test_after_f1s)/len(test_after_f1s):.2f}%')

    # ── Ensemble test metrics ─────────────────────────────────────────────────
    if ensemble_metrics:
        print(f'/n{"─"*60}')
        print('ENSEMBLE TEST METRICS (all folds combined)')
        print(f'{"─"*60}')
        e_acc = ensemble_metrics['accuracy']
        e_auc = ensemble_metrics.get('roc_auc', ensemble_metrics.get('roc_auc_ovr', 0))
        e_f1  = ensemble_metrics['f1_macro']
        print(f'  Acc={e_acc:.2f}%  AUC={e_auc:.4f}  F1={e_f1:.2f}%')

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    # ── 1. Training curves (3×3) ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Training Curves per Fold', fontsize=16, fontweight='bold')

    curve_cfg = [
        ('train_loss',  'Training Loss',        axes[0, 0]),
        ('val_loss',    'Validation Loss',       axes[0, 1]),
        ('train_f1',    'Training F1-Score',     axes[0, 2]),
        ('train_acc',   'Training Accuracy',     axes[1, 0]),
        ('val_acc',     'Validation Accuracy',   axes[1, 1]),
        ('val_f1',      'Validation F1-Score',   axes[1, 2]),
        ('train_auc',   'Training AUC',          axes[2, 0]),
        ('val_auc',     'Validation AUC',        axes[2, 1]),
        ('val_sens',    'Validation Sensitivity',axes[2, 2]),
    ]

    for result in fold_results:
        history  = result['history']
        fold_num = result['fold']
        for key, _, ax in curve_cfg:
            if key in history:
                ax.plot(history[key], label=f"Fold {fold_num}", alpha=0.7)

    for key, title, ax in curve_cfg:
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("/n✓ Training curves saved as 'training_curves.png'")

    # ── 2. Per-fold Test comparison (before vs after calibration) ────────────
    n_folds   = len(fold_test_results)
    fold_nums = [f + 1 for f in sorted(fold_test_results.keys())]
    has_calib = any(ftr['after_calibration'] is not None for ftr in fold_test_results.values())

    metrics_to_plot = [
        ('accuracy', 'Accuracy (%)'),
        ('auc',      'AUC'),
        ('f1_macro', 'F1-Macro (%)'),
    ]

    def _get_metric(m_dict, key):
        if key == 'auc':
            return m_dict.get('roc_auc', m_dict.get('roc_auc_ovr', 0))
        return m_dict.get(key, 0)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Per-Fold Test Metrics: Before vs After Calibration',
                  fontsize=14, fontweight='bold')

    x = range(n_folds)
    width = 0.35

    for ax, (metric_key, metric_label) in zip(axes2, metrics_to_plot):
        before_vals = [_get_metric(fold_test_results[f]['before_calibration'], metric_key)
                       for f in sorted(fold_test_results.keys())]
        bars_b = ax.bar([xi - width/2 for xi in x], before_vals, width,
                        label='Before Calib', color='steelblue', alpha=0.8)

        if has_calib:
            after_vals = [
                _get_metric(fold_test_results[f]['after_calibration'], metric_key)
                if fold_test_results[f]['after_calibration'] is not None else 0
                for f in sorted(fold_test_results.keys())
            ]
            bars_a = ax.bar([xi + width/2 for xi in x], after_vals, width,
                            label='After Calib', color='darkorange', alpha=0.8)
            for bar, val in zip(bars_a, after_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        for bar, val in zip(bars_b, before_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(list(x))
        ax.set_xticklabels([f'Fold {f}' for f in fold_nums])
        ax.set_title(metric_label)
        ax.set_xlabel('Fold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('per_fold_test_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Per-fold test metrics saved as 'per_fold_test_metrics.png'")

    # ── 3. Per-class Threshold per Fold ───────────────────────────────────────
    if has_calib:
        import numpy as np
        thresh_data = [
            fold_test_results[f]['thresholds']
            for f in sorted(fold_test_results.keys())
            if fold_test_results[f]['thresholds'] is not None
        ]
        if thresh_data:
            thresh_matrix = np.array(thresh_data)          # [n_folds, n_classes]
            n_valid_folds, n_cls = thresh_matrix.shape

            fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
            fig3.suptitle('Per-class Decision Thresholds (Beta Calibration + Threshold Tuning)',
                          fontsize=13, fontweight='bold')

            # -- Heatmap: folds × classes --
            ax_heat = axes3[0]
            im = ax_heat.imshow(thresh_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0.0, vmax=1.0)
            ax_heat.set_xticks(range(n_cls))
            ax_heat.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
            ax_heat.set_yticks(range(n_valid_folds))
            ax_heat.set_yticklabels([f'Fold {f+1}' for f in range(n_valid_folds)])
            ax_heat.set_title('Threshold Heatmap (Fold × Class)')
            ax_heat.set_xlabel('Class')
            for i in range(n_valid_folds):
                for j in range(n_cls):
                    ax_heat.text(j, i, f'{thresh_matrix[i, j]:.2f}',
                                 ha='center', va='center', fontsize=8,
                                 color='black')
            plt.colorbar(im, ax=ax_heat, label='Threshold')

            # -- Line plot: mean ± std per class --
            ax_line = axes3[1]
            mean_t  = thresh_matrix.mean(axis=0)
            std_t   = thresh_matrix.std(axis=0)
            x_cls   = range(n_cls)
            ax_line.bar(x_cls, mean_t, yerr=std_t, capsize=4,
                        color='mediumseagreen', alpha=0.85, edgecolor='black', label='Mean ± Std')
            ax_line.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Default (0.5)')
            for xi, (m, s) in enumerate(zip(mean_t, std_t)):
                ax_line.text(xi, m + s + 0.01, f'{m:.2f}', ha='center', va='bottom', fontsize=8)
            ax_line.set_xticks(list(x_cls))
            ax_line.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
            ax_line.set_title('Mean Threshold per Class (across folds)')
            ax_line.set_xlabel('Class')
            ax_line.set_ylabel('Threshold')
            ax_line.set_ylim(0.0, 1.05)
            ax_line.legend()
            ax_line.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig('thresholds_per_fold.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✓ Threshold plot saved as 'thresholds_per_fold.png'")
