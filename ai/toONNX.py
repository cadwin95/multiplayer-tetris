def save_final_results(model, optimizer, episodes, epsilon, stats, log_dir):
    """최종 결과 저장"""
    # 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episodes': episodes,
        'epsilon': epsilon,
        'stats': stats
    }, os.path.join(log_dir, 'final_model.pth'))

    # ONNX 모델로 변환
    model.eval()
    dummy_input = torch.randn(1, 607)  # 입력 크기에 맞게 조정
    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(log_dir, 'tetris_model.onnx'),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("ONNX model saved to:", os.path.join(log_dir, 'tetris_model.onnx'))

    # 학습 그래프 생성
    # ... rest of the code ... 