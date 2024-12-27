/// 이 파일에서는 게임 스키마 Object를 한번에 관리할 수 있도록 정리

import { defineSchema, defineTable } from "convex/server";    
import { v, Infer} from "convex/values";
import { Id } from "./_generated/dataModel";

// 1. 게임 관련 상수 정의
export const GAME_VALUES = {
  STATUS: ["waiting", "ready", "playing", "paused", "finished", "idle"] as const,
  MODES: ["solo", "multi", "ai"] as const,
  PIECES: ["I", "O", "T", "S", "Z", "J", "L"] as const,  
  DIRECTIONS: ["left", "right", "down", "rotate", "hardDrop", "hold"] as const,
  INITIAL_BOARD_SIZE: { WIDTH: 10, HEIGHT: 20 } as const,
  BASE_POINTS: {
    1: 100, 2: 300, 3: 500, 4: 800
  } as const
} as const;

// 모든 타입과 상수를 한 곳에서 관리
export type GameStatus = typeof GAME_VALUES.STATUS[number];
export type GameMode = typeof GAME_VALUES.MODES[number];
export type PieceType = typeof GAME_VALUES.PIECES[number];
export type DirectionType = typeof GAME_VALUES.DIRECTIONS[number];

// 상수 export
export const { PIECES: PIECE_VALUES } = GAME_VALUES;

// 2. 타입 스크립트 용 정의
export interface tsValidators { 
  STATUS: typeof GAME_VALUES.STATUS[number]; // 예시 : "waiting" | "ready" | "playing" | "paused" | "finished"
  MODES: typeof GAME_VALUES.MODES[number]; // 예시 : "solo" | "multi"
  PIECES: typeof GAME_VALUES.PIECES[number]; // 예시 : "I" | "O" | "T" | "S" | "Z" | "J" | "L"
  DIRECTIONS: typeof GAME_VALUES.DIRECTIONS[number]; // 예시 : "left" | "right" | "down" | "rotate" | "hardDrop" | "hold"
}

// 3. Convex 검증 타입 // Mutation, Query 검증 타입
export const convexValidators = {
  STATUS: v.union(...GAME_VALUES.STATUS.map(s => v.literal(s))), // 예시 : v.literal("waiting") | v.literal("ready") | v.literal("playing") | v.literal("paused") | v.literal("finished")
  MODES: v.union(...GAME_VALUES.MODES.map(m => v.literal(m))), // 예시 : v.literal("solo") | v.literal("multi")
  PIECES: v.union(...GAME_VALUES.PIECES.map(p => v.literal(p))), // 예시 : v.literal("I") | v.literal("O") | v.literal("T") | v.literal("S") | v.literal("Z") | v.literal("J") | v.literal("L")
  DIRECTIONS: v.union(...GAME_VALUES.DIRECTIONS.map(d => v.literal(d))), // 예시 : v.literal("left") | v.literal("right") | v.literal("down") | v.literal("rotate") | v.literal("hardDrop") | v.literal("hold")
};

// 4. 게임 
export const PIECE_ROTATIONS: { 
  [key in tsValidators["PIECES"]]: number[][][]
} = {   
  I: [[[1, 1, 1, 1]], [[1], [1], [1], [1]], [[1, 1, 1, 1]], [[1], [1], [1], [1]]],
  O: [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]],
  T: [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
  S: [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]], [[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
  Z: [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]], [[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]],
  J: [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
  L: [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]]
};

// 피스 인터페이스
export interface Piece {
  type: PieceType;
  rotation: number;
  position: { x: number; y: number };
}

// 1. 게임 테이블 스키마 정의
export const gameTableSchema = v.object({
  status: convexValidators.STATUS,
  mode: convexValidators.MODES,
  players: v.array(v.id("players")),
  settings: v.object({
    startLevel: v.number(),
    gameSpeed: v.number(),
  }),
  winnerId: v.optional(v.id("players")),
});

// 2. 플레이어 테이블 스키마 정의
export const playerTableSchema = v.object({
  gameId: v.union(v.id("games"), v.null()),
  playerName: v.string(),
  score: v.number(),
  level: v.number(),
  lines: v.number(),
  board: v.string(),
  currentPiece: convexValidators.PIECES,
  nextPiece: convexValidators.PIECES,
  holdPiece: v.union(convexValidators.PIECES, v.null()),
  position: v.object({
    x: v.number(),
    y: v.number()
  }),
  rotation: v.number(),
  isReady: v.boolean(),
  isPlaying: v.boolean(),
  garbageLines: v.number(),
  isAI: v.optional(v.boolean())
});

// 게임 히스토리 저장을 위한 스키마 추가
export const gameHistorySchema = v.object({
  playerId: v.id("players"),
  gameId: v.id("games"),
  sequence: v.number(),  // 게임 내에서의 액션 순서
  action: convexValidators.DIRECTIONS,
  pieceType: convexValidators.PIECES,  // 현재 조작 중인 피스
  position: v.object({
    x: v.number(),
    y: v.number()
  }),
  rotation: v.number(),
  linesCleared: v.number(),  // 이 액션으로 인해 제거된 라인 수
  board: v.string(),  // 액션 이전의 보드 상태
  nextPiece: convexValidators.PIECES,  // 다음 피스
  holdPiece: v.optional(convexValidators.PIECES),  // 홀드된 피스
  score: v.number(),  // 현재 점수
  level: v.number()  // 현재 레벨
});

// 게임 결과 요약 스키마 추가 (게임당 하나의 레코드)
export const gameResultSchema = v.object({
  gameId: v.id("games"),
  playerId: v.id("players"),
  startTime: v.number(),
  endTime: v.number(),
  finalScore: v.number(),
  totalLines: v.number(),
  maxCombo: v.number(),
  averageSpeed: v.number(),  // 평균 의사결정 시간 (밀리초)
  totalPieces: v.number(),   // 총 사용된 피스 수
});

// 3. 스키마 정의 업데이트
export default defineSchema({
  games: defineTable(gameTableSchema),
  players: defineTable(playerTableSchema).index("by_game", ["gameId"]),
  gameHistory: defineTable(gameHistorySchema)
    .index("by_game_sequence", ["gameId", "sequence"])
    .index("by_player", ["playerId"]),
  gameResults: defineTable(gameResultSchema)
    .index("by_player", ["playerId"])
    .index("by_game", ["gameId"])
});

// 4. Document 타입 유도 (ID 포함)
export type DbGameState = Infer<typeof gameTableSchema> & { _id: Id<"games"> };
export type DbPlayerState = Infer<typeof playerTableSchema> & { _id: Id<"players"> };
export type DbGameHistory = Infer<typeof gameHistorySchema> & { _id: Id<"gameHistory"> };
export type DbGameResult = Infer<typeof gameResultSchema> & { _id: Id<"gameResults"> };

// 클라이언트용 게임 상태 인터페이스
export interface ClientGameState {
  board: string;
  currentPiece: Piece;
  nextPiece: Piece;
  holdPiece: Piece | null;
  score: number;
  level: number;
  lines: number;
  status: GameStatus;
}

// 클라이언트 타입
export interface Piece {
  type: PieceType;
  rotation: number;
  position: { x: number; y: number };
}

export const PIECE_COLORS: { [key in PieceType]: string } = {
    I: "cyan",
    O: "yellow",
    T: "purple",
    S: "green",
    Z: "red",
    J: "blue",
    L: "orange"
} as const;