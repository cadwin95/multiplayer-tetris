// games.ts
import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { Id } from "./_generated/dataModel";
import {  MutationCtx } from "./_generated/server";

const TETROMINOS = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
const BASE_POINTS = [0, 40, 100, 300, 1200]; // 0, 1, 2, 3, 4 라인 제거시 점수
const GARBAGE_LINE_POINTS = 50; // 방해 라인 보내기에 필요한 점수
const getRandomPiece = () => TETROMINOS[Math.floor(Math.random() * TETROMINOS.length)] + '0';

// 게임 생성
export const createGame = mutation({
  args: {
    playerName: v.string(),
  },
  handler: async (ctx, args) => {
    const gameId = await ctx.db.insert("games", {
      status: "waiting",
      players: [],
      settings: {
        startLevel: 1,
        gameSpeed: 1000,
      },
      winnerId: null
    });

    const playerId = await ctx.db.insert("players", {
      gameId,
      name: args.playerName,
      score: 0,
      level: 1,
      lines: 0,
      board: "0".repeat(200),
      currentPiece: getRandomPiece(),
      nextPiece: getRandomPiece(),
      position: { x: 4, y: 0 },
      isReady: false,
      isPlaying: false,
      garbageLines: 0,
    });

    await ctx.db.patch(gameId, {
      players: [playerId],
    });

    return { gameId, playerId };
  },
});

// 게임 참가
export const joinGame = mutation({
  args: {
    gameId: v.id("games"),
    playerName: v.string(),
  },
  handler: async (ctx, args) => {
    const game = await ctx.db.get(args.gameId);
    if (!game || game.status !== "waiting") throw new Error("Cannot join game");

    const playerId = await ctx.db.insert("players", {
      gameId: args.gameId,
      name: args.playerName,
      score: 0,
      level: 1,
      lines: 0,
      board: "0".repeat(200),
      currentPiece: getRandomPiece(),
      nextPiece: getRandomPiece(),
      position: { x: 4, y: 0 },
      isReady: false,
      isPlaying: false,
      garbageLines: 0,
    });

    await ctx.db.patch(args.gameId, {
      players: [...game.players, playerId],
    });

    return playerId;
  },
});

// 게임 상태 쿼리
export const getGame = query({
  args: { gameId: v.id("games") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.gameId);
  },
});

export const getPlayer = query({
  args: { playerId: v.id("players") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.playerId);
  },
});

export const getPlayers = query({
  args: { gameId: v.id("games") },
  handler: async (ctx, args) => {
    const game = await ctx.db.get(args.gameId);
    if (!game) return null;
    
    return await Promise.all(
      game.players.map((playerId: Id<"players">) => ctx.db.get(playerId))
    );
  },
});

// 테트로미노 이동
export const moveTetrominoe = mutation({
  args: {
    gameId: v.id("games"),
    playerId: v.id("players"),
    direction: v.string()
  },
  handler: async (ctx, args) => {
    const player = await ctx.db.get(args.playerId);
    if (!player || !player.isPlaying) return;

    let newX = player.position.x;
    let newY = player.position.y;

    switch (args.direction) {
      case "left":
        newX = Math.max(0, newX - 1);
        break;
      case "right":
        newX = Math.min(9, newX + 1);
        break;
      case "down":
        newY = Math.min(19, newY + 1);
        break;
    }

    const board = stringToBoard(player.board);
    const canMove = !checkCollision(board, player.currentPiece, newX, newY);

    if (canMove) {
      await ctx.db.patch(args.playerId, {
        position: { x: newX, y: newY }
      });
    } else if (args.direction === "down") {
      // 바닥에 닿았을 때 처리
      await lockPiece(ctx, args.playerId, player);
    }
  },
});

// 테트로미노 회전
export const rotateTetrominoe = mutation({
  args: {
    gameId: v.id("games"),
    playerId: v.id("players")
  },
  handler: async (ctx, args) => {
    const player = await ctx.db.get(args.playerId);
    if (!player || !player.isPlaying) return;

    const board = stringToBoard(player.board);
    const rotatedPiece = rotatePiece(player.currentPiece);

    if (!checkCollision(board, rotatedPiece, player.position.x, player.position.y)) {
      await ctx.db.patch(args.playerId, {
        currentPiece: rotatedPiece
      });
    }
  },
});

// 하드 드롭
export const hardDrop = mutation({
  args: { 
    gameId: v.id("games"), 
    playerId: v.id("players") 
  },
  handler: async (ctx, args) => {
    const player = await ctx.db.get(args.playerId);
    if (!player || !player.isPlaying) return;

    const board = stringToBoard(player.board);
    let newY = player.position.y;

    // 충돌할 때까지 아래로 이동
    while (!checkCollision(board, player.currentPiece, player.position.x, newY + 1)) {
      newY++;
    }

    // 위치 업데이트 후 조각 고정
    await ctx.db.patch(args.playerId, {
      position: { x: player.position.x, y: newY }
    });
    
    await lockPiece(ctx, args.playerId, {
      ...player,
      position: { x: player.position.x, y: newY }
    });
  },
});

// 자동 하강
export const autoDown = mutation({
  args: {
    gameId: v.id("games"),
    playerId: v.id("players")
  },
  handler: async (ctx, args) => {
    const player = await ctx.db.get(args.playerId);
    if (!player || !player.isPlaying) return;

    // Call moveTetrominoe mutation directly
    await moveTetrominoe(ctx, {
      gameId: args.gameId,
      playerId: args.playerId,
      direction: "down"
    });
  }
});

// 게임 시작
export const startGame = mutation({
  args: {
    gameId: v.id("games")
  },
  handler: async (ctx, args) => {
    const game = await ctx.db.get(args.gameId);
    if (!game) return;

    await ctx.db.patch(args.gameId, {
      status: "playing"
    });

    const players = await ctx.db
      .query("players")
      .filter(q => q.eq(q.field("gameId"), args.gameId))
      .collect();

    for (const player of players) {
      await ctx.db.patch(player._id, {
        isPlaying: true,
        board: "0".repeat(200),
        score: 0,
        level: 1,
        lines: 0,
        currentPiece: getRandomPiece(),
        nextPiece: getRandomPiece(),
        position: { x: 4, y: 0 },
        garbageLines: 0
      });
    }
  }
});

// 유틸리티 함수들
function stringToBoard(boardString: string): number[][] {
  const board = [];
  for (let i = 0; i < 20; i++) {
    board.push(boardString.slice(i * 10, (i + 1) * 10).split('').map(Number));
  }
  return board;
}

function boardToString(board: number[][]): string {
  return board.flat().join('');
}

async function lockPiece(
  ctx: MutationCtx, 
  playerId: Id<"players">, 
  player: {
    _id: Id<"players">;
    gameId: Id<"games">;
    board: string;
    currentPiece: string;
    nextPiece: string;
    position: { x: number; y: number };
    score: number;
    level: number;
    lines: number;
    isPlaying: boolean;
    garbageLines: number;
  }
) {
  const board = stringToBoard(player.board);
  const updatedBoard = placePiece(board, player.currentPiece, player.position.x, player.position.y);
  
  // 라인 제거 처리
  const { updatedBoard: clearedBoard, linesCleared } = clearLines(updatedBoard);
  
  // 점수 계산
  const additionalScore = calculateScore(linesCleared, player.level);
  const newLines = player.lines + linesCleared;
  const newLevel = Math.floor(newLines / 10) + 1;
  
  // 방해 라인 처리
  let garbageLines = player.garbageLines;
  if (linesCleared > 0) {
    garbageLines = Math.max(0, garbageLines - linesCleared);
  }
  
  // 게임 오버 체크
  if (checkGameOver(clearedBoard)) {
    await handleGameOver(ctx, player.gameId, playerId);
    return;
  }

  // 상태 업데이트
  await ctx.db.patch(playerId, {
    board: boardToString(clearedBoard),
    score: player.score + additionalScore,
    level: newLevel,
    lines: newLines,
    currentPiece: player.nextPiece,
    nextPiece: getRandomPiece(),
    position: { x: 4, y: 0 },
    garbageLines
  });

  // 방해 라인 보내기 체크
  if (additionalScore >= GARBAGE_LINE_POINTS) {
    await sendGarbageLines(ctx, player.gameId, playerId);
  }
}

function checkCollision(board: number[][], piece: string, x: number, y: number): boolean {
  const pieceMatrix = getPieceMatrix(piece);
  
  for (let py = 0; py < pieceMatrix.length; py++) {
    for (let px = 0; px < pieceMatrix[py].length; px++) {
      if (pieceMatrix[py][px]) {
        const boardX = x + px;
        const boardY = y + py;

        if (
          boardX < 0 || 
          boardX >= 10 || 
          boardY >= 20 ||
          (boardY >= 0 && board[boardY][boardX])
        ) {
          return true;
        }
      }
    }
  }
  return false;
}

function placePiece(board: number[][], piece: string, x: number, y: number): number[][] {
  const pieceMatrix = getPieceMatrix(piece);
  const newBoard = board.map(row => [...row]);

  for (let py = 0; py < pieceMatrix.length; py++) {
    for (let px = 0; px < pieceMatrix[py].length; px++) {
      if (pieceMatrix[py][px] && y + py >= 0) {
        newBoard[y + py][x + px] = 1;
      }
    }
  }

  return newBoard;
}

function clearLines(board: number[][]): { updatedBoard: number[][], linesCleared: number } {
  const newBoard = board.filter(row => !row.every(cell => cell === 1));
  const linesCleared = 20 - newBoard.length;
  
  while (newBoard.length < 20) {
    newBoard.unshift(Array(10).fill(0));
  }
  
  return {
    updatedBoard: newBoard,
    linesCleared
  };
}

function calculateScore(linesCleared: number, level: number): number {
  return BASE_POINTS[linesCleared] * level;
}

function checkGameOver(board: number[][]): boolean {
  return board[0].some(cell => cell === 1);
}

async function handleGameOver(ctx: MutationCtx, gameId: Id<"games">, loserId: Id<"players">) {
  const game = await ctx.db.get(gameId);
  if (!game) return;

  // 승자 결정 (2인용 게임 기준)
  const winnerId = game.players.find((id: Id<"players">) => id !== loserId);

  await ctx.db.patch(gameId, {
    status: "finished",
    winnerId
  });

  // 모든 플레이어 게임 종료 처리
  for (const playerId of game.players) {
    await ctx.db.patch(playerId, {
      isPlaying: false
    });
  }
}

async function sendGarbageLines(ctx: MutationCtx, gameId: Id<"games">, fromPlayerId: Id<"players">) {
  const game = await ctx.db.get(gameId);
  if (!game) return;

  // 상대 플레이어에게 방해 라인 추가
  const targetPlayerId = game.players.find((id: Id<"players">) => id !== fromPlayerId);
  if (!targetPlayerId) return;

  const targetPlayer = await ctx.db.get(targetPlayerId);
  if (!targetPlayer) return;

  await ctx.db.patch(targetPlayerId, {
    garbageLines: targetPlayer.garbageLines + 1
  });
}

function getPieceMatrix(piece: string): number[][] {
  const shapes: { [key: string]: number[][] } = {
    'I0': [[1, 1, 1, 1]],
    'I1': [[1], [1], [1], [1]],
    'O': [[1, 1], [1, 1]],
    'T0': [[0, 1, 0], [1, 1, 1]],
    'T1': [[1, 0], [1, 1], [1, 0]],
    'T2': [[1, 1, 1], [0, 1, 0]],
    'T3': [[0, 1], [1, 1], [0, 1]],
    'S0': [[0, 1, 1], [1, 1, 0]],
    'S1': [[1, 0], [1, 1], [0, 1]],
    'Z0': [[1, 1, 0], [0, 1, 1]],
    'Z1': [[0, 1], [1, 1], [1, 0]],
    'J0': [[1, 0, 0], [1, 1, 1]],
    'J1': [[1, 1], [1, 0], [1, 0]],
    'J2': [[1, 1, 1], [0, 0, 1]],
    'J3': [[0, 1], [0, 1], [1, 1]],
    'L0': [[0, 0, 1], [1, 1, 1]],
    'L1': [[1, 0], [1, 0], [1, 1]],
    'L2': [[1, 1, 1], [1, 0, 0]],
    'L3': [[1, 1], [0, 1], [0, 1]]
  };
  return shapes[piece] || [[1]];
}

function rotatePiece(piece: string): string {
  const rotations: { [key: string]: string } = {
    'I0': 'I1', 'I1': 'I0',
    'T0': 'T1', 'T1': 'T2', 'T2': 'T3', 'T3': 'T0',
    'S0': 'S1', 'S1': 'S0',
    'Z0': 'Z1', 'Z1': 'Z0',
    'J0': 'J1', 'J1': 'J2', 'J2': 'J3', 'J3': 'J0',
    'L0': 'L1', 'L1': 'L2', 'L2': 'L3', 'L3': 'L0',
    'O': 'O'
  };
  return rotations[piece] || piece;
}

// 게임 일시정지
export const pauseGame = mutation({
  args: { gameId: v.id("games") },
  handler: async (ctx, args) => {
    const game = await ctx.db.get(args.gameId);
    if (!game) return;

    const newStatus = game.status === "playing" ? "paused" : "playing";
    
    await ctx.db.patch(args.gameId, {
      status: newStatus
    });

    const players = await Promise.all(
      game.players.map((id: Id<"players">) => ctx.db.get(id))
    );
      
    for (const player of players) {
      if (player) {
        await ctx.db.patch(player._id, {
          isPlaying: newStatus === "playing"
        });
      }
    }
  }
});

// Ready 상태 설정
export const setReady = mutation({
  args: {
    gameId: v.id("games"),
    playerId: v.id("players")
  },
  handler: async (ctx, args) => {
    const player = await ctx.db.get(args.playerId);
    if (!player) return;

    await ctx.db.patch(args.playerId, {
      isReady: true
    });

    // 모든 플레이어가 준비되었는지 확인
    const game = await ctx.db.get(args.gameId);
    if (!game) return;

    const allPlayers = await Promise.all(
      game.players.map((pid: Id<"players">) => ctx.db.get(pid))
    );

    const allReady = allPlayers.every(p => p?.isReady);
    if (allReady) {
      // Call startGame mutation directly
      await startGame(ctx, { gameId: args.gameId });
    }
  }
});

// Add this query at the top with other queries
export const listGames = query({
  handler: async (ctx) => {
    return await ctx.db
      .query("games")
      .filter(q => q.neq(q.field("status"), "finished"))
      .collect();
  },
});