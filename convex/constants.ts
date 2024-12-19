// 게임 관련 상수들을 별도 파일로 분리
export const GAME_CONSTANTS = {
  BOARD: {
    WIDTH: 10,
    HEIGHT: 20,
    SPAWN_POSITION: { x: 4, y: 0 }
  },
  SCORING: {
    BASE_POINTS: {
      1: 100,  // Single
      2: 300,  // Double
      3: 500,  // Triple
      4: 800   // Tetris
    },
    LEVEL_SPEEDS: {
      1: 1000,
      2: 850,
      3: 700,
      // ... 레벨별 속도
    }
  }
} as const; 