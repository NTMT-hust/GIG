import pandas as pd
import torch
import torch.nn.functional as F

class GuidedIntegratedGradients:
    def __init__(self, model, steps=64):
        self.model = model
        self.steps = steps
        self.gb = GuidedBackprop(model)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        baseline = torch.zeros_like(input_tensor)

        grads = []

        for i in range(self.steps + 1):
            xi = baseline + (i / self.steps) * (input_tensor - baseline)
            xi = xi.clone().detach().requires_grad_(True)

            out = self.model(xi)
            score = out[0, class_idx]

            self.model.zero_grad()
            score.backward()

            grads.append(xi.grad.detach())

        avg_grads = torch.stack(grads).mean(dim=0)

        attr = (input_tensor - baseline) * avg_grads   # (1,C,H,W)

        # ===== convert to heatmap =====
        heatmap_raw = attr.abs().sum(dim=1).squeeze()  # (H,W)

        heatmap = torch.clamp(heatmap_raw, min=0)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.detach().cpu().numpy(), class_idx, heatmap_raw.detach().cpu().numpy()

    def remove_hooks(self):
        self.gb.remove()
            
    def upsample_heatmap(self, heatmap, input_tensor):
        heatmap = torch.tensor(heatmap)

        heatmap = heatmap.unsqueeze(0).unsqueeze(0)

        heatmap = F.interpolate(
            heatmap,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        return heatmap.squeeze()
    
    
def save_gene_csv(
    attr,
    gene_dfs,          # list of (df, omics_type)
    output_csv,
    fold,
    sample_id,
    pred_label,
    true_label
):
    """
    attr: Tensor shape (1,3,H,W)
    gene_dfs: [(df_mrna,"mRNA"), (df_methyl,"Methylation"), (df_cnv,"CNV")]
    """

    # ===== 1. Chuẩn hóa attribution =====
    if not isinstance(attr, torch.Tensor):
        raise ValueError("attr must be a torch Tensor")

    attr = attr.detach().cpu()   # an toàn GPU
    attr = attr[0]                    # (3,H,W)

    # ===== 2. Map omics → channel =====
    omics_to_channel = {
        "mRNA": 0,
        "Methylation": 1,
        "CNV": 2
    }

    rows = []

    # ===== 3. Duyệt từng omics =====
    for df, omics in gene_dfs:

        if omics not in omics_to_channel:
            raise ValueError(f"Unknown omics type: {omics}")

        ch = omics_to_channel[omics]
        gene_map = attr[ch]  # (H,W)

        for _, r in df.iterrows():
            pixel_x = int(r["pixel_x"]) - 1
            pixel_y = int(r["pixel_y"]) - 1
            H, W = gene_map.shape

            if pixel_x >= H or pixel_y >= W:
                print("Out-of-bound gene:", r["gene_name"], pixel_x, pixel_y)


            score = gene_map[pixel_x, pixel_y].item()

            rows.append([
                fold,
                sample_id,
                pred_label,
                true_label,
                pixel_x,
                pixel_y,
                r["gene_name"],
                omics,
                score
            ])

    # ===== 4. Save CSV =====
    out_df = pd.DataFrame(
        rows,
        columns=[
            "fold",
            "sample_id",
            "predicted_label",
            "true_label",
            "row",
            "col",
            "gene_name",
            "omics_type",
            "attribute_score"
        ]
    )

    out_df.to_csv(output_csv, index=False)


class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def relu_backward_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        for m in self.model.modules():
            if isinstance(m, torch.nn.ReLU):
                h = m.register_full_backward_hook(relu_backward_hook)
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()