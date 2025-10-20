"""
# @file name  : multi-task model.py
# @author     : Ziyao Ling
# @date       : 01/12/2024
# @brief      : multi-task mobilenet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict


class MobileNetMultiTask(nn.Module):
    """MobileNet Multitask"""

    def __init__(self,
                 num_classes_dict: Dict[str, int],
                 model_name: str = 'mobilenetv3_large_100',
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):

        super(MobileNetMultiTask, self).__init__()

        self.num_classes_dict = num_classes_dict
        self.model_name = model_name

        # 创建backbone
        print(f"Loading MobileNet model: {model_name}")
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # 移除最后的分类层
            )
            print(f"✅ Successfully loaded {model_name}")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            print("Available MobileNet models in timm:")
            available_models = [m for m in timm.list_models() if 'mobile' in m.lower()]
            for m in available_models[:10]:  # 显示前10个
                print(f"  - {m}")
            raise

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                self.backbone = self.backbone.cuda()
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
            if torch.cuda.is_available():
                self.backbone = self.backbone.cpu()

        print(f"Detected feature dimension: {feature_dim}")

        # 共享特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7)
        )

        # 任务特定的分类头
        self.dynasty_head = nn.Sequential(
            nn.Linear(256, num_classes_dict['dynasty'])
        )

        self.kiln_head = nn.Sequential(
            nn.Linear(256, num_classes_dict['kiln'])
        )

        self.glaze_head = nn.Sequential(
            nn.Linear(256, num_classes_dict['glaze'])
        )

        self.pattern_head = nn.Sequential(
            nn.Linear(256, num_classes_dict['pattern'])
        )

        self._initialize_heads()

    def _initialize_heads(self):
        for module in [self.dynasty_head, self.kiln_head,
                       self.glaze_head, self.pattern_head, self.vessel_type_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x):

        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        shared_features = self.feature_projection(features)
        dynasty_out = self.dynasty_head(shared_features)
        kiln_out = self.kiln_head(shared_features)
        glaze_out = self.glaze_head(shared_features)
        pattern_out = self.pattern_head(shared_features)
        type_out = self.vessel_type_head(shared_features)

        return {
            'dynasty': dynasty_out,
            'kiln': kiln_out,
            'glaze': glaze_out,
            'pattern': pattern_out,
            'type': type_out
        }

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_head_params(self):
        params = []
        params.extend(self.feature_projection.parameters())
        params.extend(self.dynasty_head.parameters())
        params.extend(self.kiln_head.parameters())
        params.extend(self.glaze_head.parameters())
        params.extend(self.pattern_head.parameters())
        params.extend(self.vessel_type_head.parameters())
        return params

    def freeze_backbone(self, freeze=True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'heads': sum(p.numel() for p in self.get_head_params())
        }


def test_mobilenet_models():
    models_to_test = [
        'mobilenetv3_large_100',
        'mobilenetv3_small_100',
        'mobilenetv2_100',
    ]

    num_classes = {
        'dynasty': 2,
        'kiln': 17,
        'glaze': 17,
        'pattern': 6,
        'type': 20
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"Testing {model_name}...")

        try:
            model = MobileNetMultiTask(
                num_classes_dict=num_classes,
                model_name=model_name,
                pretrained=True
            ).to(device)
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            outputs = model(dummy_input)
            print(f"✅ Model loaded successfully!")
            print(f"Output shapes:")
            for task, output in outputs.items():
                print(f"  {task}: {output.shape}")
            params_info = model.get_num_params()
            print(f"\nParameter count:")
            print(f"  Total: {params_info['total']:,}")
            print(f"  Backbone: {params_info['backbone']:,}")
            print(f"  Heads: {params_info['heads']:,}")
            print(f"  Ratio: {params_info['backbone'] / params_info['total'] * 100:.1f}% backbone")
            print(f"\nTesting different input sizes:")
            for size in [224, 256, 192]:
                try:
                    test_input = torch.randn(1, 3, size, size).to(device)
                    _ = model(test_input)
                    print(f"  {size}x{size}: ✅")
                except Exception as e:
                    print(f"  {size}x{size}: ❌ {str(e)}")

        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")

        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Testing complete!")

def create_mobilenet_optimizer(model, config):
    if config.get('differential_lr', True):
        optimizer = torch.optim.AdamW([
            {'params': model.get_backbone_params(),
             'lr': config.get('lr_backbone', 1e-4),
             'weight_decay': config.get('weight_decay', 1e-4)},
            {'params': model.get_head_params(),
             'lr': config.get('lr_head', 1e-3),
             'weight_decay': config.get('weight_decay', 1e-4)}
        ])
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )

    return optimizer


if __name__ == "__main__":

    print("Testing MobileNet Multi-Task Models...")
    test_mobilenet_models()

    print("\n" + "=" * 60)
    print("Creating example model for training...")

    model = MobileNetMultiTask(
        num_classes_dict={'dynasty': 2, 'kiln': 17, 'glaze': 17, 'pattern': 6, 'type': 20},
        model_name='mobilenetv3_large_100',
        pretrained=True
    )

    print(f"\nModel structure:")
    print(f"- Backbone: {model.model_name}")
    print(f"- Shared projection: {model.feature_projection}")
    print(f"- Task heads: 5 (dynasty, kiln, glaze, pattern, type)")

    params = model.get_num_params()
    print(f"\nTotal parameters: {params['total']:,}")