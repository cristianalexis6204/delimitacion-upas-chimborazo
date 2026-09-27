"""
===================================================================================================
MÓDULO DE SUPER-RESOLUCIÓN POR APRENDIZAJE PROFUNDO (SINGLE IMAGE SUPER-RESOLUTION - SISR 2X)
Memoria de Trabajo Fin de Máster (TFM) - UNIR
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Implementa la arquitectura EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution)
optimizada en PyTorch CUDA para duplicar la resolución espacial (0.30 m/px -> 0.15 m/px)
preservando la fidelidad radiométrica y geométrica sin alucinaciones de artefactos GAN.
===================================================================================================
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Bloque Residual Básico de EDSR sin capas BatchNorm para evitar compresión de rango dinámico."""
    def __init__(self, n_feats, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale

class EDSR(nn.Module):
    """Arquitectura canónica EDSR (Lim et al., CVPR 2017) para Super-Resolución 2x."""
    def __init__(self, n_feats=64, n_resblocks=16, scale=2, res_scale=1.0):
        super().__init__()
        # Normalización radiométrica RGB
        self.sub_mean = nn.Conv2d(3, 3, 1)
        self.add_mean = nn.Conv2d(3, 3, 1)
        
        # Extracción inicial de características (Head)
        self.head = nn.Sequential(nn.Conv2d(3, n_feats, 3, padding=1))
        
        # Tronco residual profundo (Body)
        body_modules = [ResBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        body_modules.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*body_modules)
        
        # Módulo de reconstrucción y upsampling por PixelShuffle (Tail)
        self.tail = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(n_feats, n_feats * (scale ** 2), 3, padding=1),
                nn.PixelShuffle(scale)
            ),
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

class DeepSuperResolutionEngine:
    """Motor de inferencia de Super-Resolución satelital con soporte GPU y teselado suave."""
    def __init__(self, model_path=None, device=None, tile_size=256, tile_overlap=32):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.scale = 2
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "data", "models", "edsr_baseline_2x.pt")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el checkpoint EDSR 2x en: {model_path}")
            
        print(f"[DeepSR] Inicializando EDSR 2x en dispositivo: {self.device}")
        self.model = EDSR(n_feats=64, n_resblocks=16, scale=self.scale)
        ckpt = torch.load(model_path, map_location='cpu')
        clean_sd = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(clean_sd)
        self.model.to(self.device)
        self.model.eval()

    def super_resolve(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Aumenta al doble la resolución de una imagen RGB (H, W, 3) -> (2*H, 2*W, 3).
        Utiliza teselado con ventanas solapadas para garantizar estabilidad de VRAM en GPU.
        """
        h, w, c = img_rgb.shape
        assert c == 3, f"La imagen debe ser RGB de 3 canales, se recibió shape: {img_rgb.shape}"
        
        # Si la imagen es moderada (<= 512x512), procesar directamente en un solo pase
        if h <= 512 and w <= 512:
            return self._process_single_patch(img_rgb)
            
        # Para escenas grandes, aplicar teselado con ventana deslizante y solapamiento
        out_h, out_w = h * self.scale, w * self.scale
        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)
        
        step = self.tile_size - self.tile_overlap
        y_starts = list(range(0, h - self.tile_size + 1, step))
        if y_starts[-1] + self.tile_size < h:
            y_starts.append(h - self.tile_size)
            
        x_starts = list(range(0, w - self.tile_size + 1, step))
        if x_starts[-1] + self.tile_size < w:
            x_starts.append(w - self.tile_size)
            
        # Ponderación hanning para fusión suave de bordes
        han_y = np.hanning(self.tile_size * self.scale)
        han_x = np.hanning(self.tile_size * self.scale)
        tile_weight = np.outer(han_y, han_x)[:, :, np.newaxis].astype(np.float32) + 1e-4

        for y in y_starts:
            for x in x_starts:
                patch = img_rgb[y:y + self.tile_size, x:x + self.tile_size]
                sr_patch = self._process_single_patch(patch)
                
                oy, ox = y * self.scale, x * self.scale
                oh, ow = self.tile_size * self.scale, self.tile_size * self.scale
                
                output[oy:oy + oh, ox:ox + ow] += sr_patch.astype(np.float32) * tile_weight
                weight_map[oy:oy + oh, ox:ox + ow] += tile_weight

        final_out = np.clip(output / weight_map, 0, 255).astype(np.uint8)
        return final_out

    def _process_single_patch(self, patch_rgb: np.ndarray) -> np.ndarray:
        """Inferencia directa en GPU de un parche individual."""
        # Convertir a tensor [1, 3, H, W] en rango [0, 255] (EDSR fue entrenado con este rango)
        tensor = torch.from_numpy(patch_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            sr_tensor = self.model(tensor)
            
        sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(sr_np, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    engine = DeepSuperResolutionEngine()
    test_img = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    sr_res = engine.super_resolve(test_img)
    print(f"[OK] Test Super-Resolución: {test_img.shape} -> {sr_res.shape}")
