import torch
from model import DQN

def convert_to_onnx(model_path: str, output_path: str):
    # 모델 로드
    model = DQN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 더미 입력 생성 (상태 크기에 맞춰서)
    dummy_input = torch.randn(1, 200 + 16 + 16 + 10 + 10 + 1)  # 보드 + next + hold + heights + holes + can_hold

    # ONNX로 변환
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )

if __name__ == "__main__":
    convert_to_onnx("tetris_model.pth", "tetris_model.onnx") 