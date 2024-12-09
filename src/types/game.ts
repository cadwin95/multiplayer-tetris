// src/types/tetris.ts
export type Piece = number[][];
export type Board = number[][];

export interface GameState {
  board: string;
  currentPiece: string;
  position: { x: number; y: number };
  nextPiece: string;
  score: number;
  level: number;
  lines: number;
}

export interface Player {
  _id: string;
  name: string;
  gameState: GameState;
}

export const PIECES: Piece[] = [
  // I piece
  [[1, 1, 1, 1]],
  // L piece
  [
    [1, 0, 0],
    [1, 1, 1]
  ],
  // J piece
  [
    [0, 0, 1],
    [1, 1, 1]
  ],
  // O piece
  [
    [1, 1],
    [1, 1]
  ],
  // S piece
  [
    [0, 1, 1],
    [1, 1, 0]
  ],
  // T piece
  [
    [0, 1, 0],
    [1, 1, 1]
  ],
  // Z piece
  [
    [1, 1, 0],
    [0, 1, 1]
  ]
];

export const BOARD_WIDTH = 10;
export const BOARD_HEIGHT = 20;

export interface LocalGameState {
  board: string;
  currentPiece: string;
  position: { x: number; y: number };
  nextPiece: string;
  score: number;
  level: number;
  lines: number;
  isValid?: boolean;
}