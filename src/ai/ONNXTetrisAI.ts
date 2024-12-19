import { ClientGameState, DirectionType } from '../../convex/schema';
import { InferenceSession, Tensor } from 'onnxruntime-web';

export class ONNXTetrisAI {
  private session: InferenceSession | null = null;
  private readonly actionMap: DirectionType[] = ['left', 'right', 'rotate', 'down', 'hardDrop', 'hold'];

  constructor() {
    this.initModel();
  }

  private async initModel() {
    try {
      this.session = await InferenceSession.create('/tetris_model.onnx');
    } catch (e) {
      console.error('Failed to load ONNX model:', e);
    }
  }

  private prepareState(gameState: Partial<ClientGameState>): Float32Array {
    if (!gameState.board) throw new Error('Board is required');
    
    // 게임 상태를 모델 입력 형식으로 변환
    const board = new Float32Array(200);  // 20x10 보드
    const nextPiece = new Float32Array(16);  // 4x4 다음 피스
    const holdPiece = new Float32Array(16);  // 4x4 홀드 피스
    const heights = new Float32Array(10);  // 각 열의 높이
    const holes = new Float32Array(10);    // 각 열의 구멍 수
    const canHold = new Float32Array([1]); // 홀드 가능 여부

    // 보드 상태 변환
    for (let i = 0; i < gameState.board.length; i++) {
      board[i] = gameState.board[i] === '1' ? 1 : 0;
    }

    // 높이와 구멍 계산
    for (let x = 0; x < 10; x++) {
      let foundBlock = false;
      let height = 0;
      let holeCount = 0;

      for (let y = 0; y < 20; y++) {
        const cell = gameState.board[y * 10 + x];
        if (cell === '1') {
          foundBlock = true;
          height = Math.max(height, 20 - y);
        } else if (foundBlock) {
          holeCount++;
        }
      }

      heights[x] = height;
      holes[x] = holeCount;
    }

    // 모든 상태를 하나의 배열로 결합
    return Float32Array.from([
      ...board,
      ...nextPiece,
      ...holdPiece,
      ...heights,
      ...holes,
      ...canHold
    ]);
  }

  public async predictNextMove(gameState: Partial<ClientGameState>): Promise<DirectionType> {
    if (!this.session) {
      console.error('Model not loaded');
      return 'down';
    }

    try {
      if (!gameState.board) {
        throw new Error('Board is required');
      }

      // 상태 준비
      const inputTensor = new Tensor(
        'float32',
        this.prepareState(gameState),
        [1, 253]
      );

      // 추론 실행
      const output = await this.session.run({ input: inputTensor });
      const scores = Array.from(output.output.data as Float32Array);

      // 최적의 액션 선택
      const bestActionIndex = scores.indexOf(Math.max(...scores));
      return this.actionMap[bestActionIndex];

    } catch (e) {
      console.error('Prediction failed:', e);
      return 'down';
    }
  }
} 