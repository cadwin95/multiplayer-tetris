/**
 * Game Schema Exports:
 * 
 * Constants:
 * - GAME_VALUES: 게임 관련 상수 (STATUS, MODES, PIECES, DIRECTIONS, INITIAL_BOARD_SIZE, BASE_POINTS)
 * - PIECE_ROTATIONS: 각 테트리스 조각의 회전 패턴
 * 
 * TypeScript:
 * - tsValidators: 타입스크립트 타입 정의 (STATUS, MODES, PIECES, DIRECTIONS)
 * - GameState: 게임 테이블 문서 타입 (ID 포함)
 * - PlayerState: 플레이어 테이블 문서 타입 (ID 포함)
 * 
 * Convex:
 * - convexValidators: Convex 검증을 위한 validator 객체
 * - gameTableSchema: 게임 테이블 스키마
 * - playerTableSchema: 플레이어 테이블 스키마
 */

// games.ts
import { mutation, query, MutationCtx } from "./_generated/server";
import { v } from "convex/values";
import { Id } from "./_generated/dataModel";
import {    
  tsValidators,   
  GAME_VALUES,
  convexValidators, 
  PIECE_ROTATIONS, 
  DbPlayerState
} from "./schema";


// 랜덤 테트로미노 생성 함수
export const getRandomPiece = () => {
  return GAME_VALUES.PIECES[Math.floor(Math.random() * GAME_VALUES.PIECES.length)] as tsValidators["PIECES"];
}

export const createPlayer = mutation({
  args: {
    playerName: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("players", {
      playerName: args.playerName,
      score: 0,
      level: 1,
      lines: 0,
      board: "0".repeat(GAME_VALUES.INITIAL_BOARD_SIZE.WIDTH * GAME_VALUES.INITIAL_BOARD_SIZE.HEIGHT),
      currentPiece: getRandomPiece(),
      nextPiece: getRandomPiece(),
      position: { x: 4, y: 0 },
      rotation: 0,
      isPlaying: false,
      isReady: false,
      garbageLines: 0
    });
  }
});

// 게임 생성
export const createGame = mutation({
  args: {
    playerId: v.id("players"),
    mode: convexValidators.MODES
  },
  handler: async (ctx, args) => {
    try {
      const gameId = await ctx.db.insert("games", {
        status: 'waiting',
        mode: args.mode,
        players: [args.playerId],
        settings: {
          startLevel: 1,
          gameSpeed: 1000,
        },
        winnerId: undefined
      });

      return { gameId, playerId: args.playerId };
    } catch (error) {
      console.error('Error in createGame:', error);
      throw error;
    }
  },
});

// 게임 상태 쿼리
export const getGame = query({
  args: { gameId: v.id("games") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.gameId);
  },
});

// 게임 참가
export const joinGame = mutation({
  args: {
    gameId: v.id("games"),
    playerId: v.id("players"),
  },
  handler: async (ctx, args) => {
    const game = await getGame(ctx, { gameId: args.gameId });
    if (!game) throw new Error("Game not found");
    if (game.status !== "waiting") throw new Error("Game is not waiting");
    if(game.players.includes(args.playerId)) throw new Error("Player already in game");

    await ctx.db.patch(args.gameId, {
      players: [...game.players, args.playerId],
    });

    return game;
  }
});

export const getPlayer = query({
  args: { playerId: v.id("players") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.playerId);
  },
});

export const getPlayerIds = query({
  args: { gameId: v.id("games") },
  handler: async (ctx, args) => {
    const game = await ctx.db.get(args.gameId);
    if (!game) return null;
    return await Promise.all(game.players.map((playerId: Id<"players">) => ctx.db.get(playerId)));
  },
});

// 게임 히스토리 저장 함수 수정
async function _saveGameHistory(
  ctx: MutationCtx,
  args: {
    gameId: Id<"games">,
    playerId: Id<"players">,
    action: tsValidators["DIRECTIONS"],
    beforeState: {
      board: string,
      currentPiece: tsValidators["PIECES"],
      position: { x: number, y: number },
      rotation: number,
      nextPiece: tsValidators["PIECES"],
      holdPiece?: tsValidators["PIECES"],
      score: number,
      level: number,
      lines: number
    },
    afterState?: {
      board: string,
      currentPiece: tsValidators["PIECES"],
      position: { x: number, y: number },
      rotation: number,
      nextPiece: tsValidators["PIECES"],
      holdPiece?: tsValidators["PIECES"],
      score: number,
      level: number,
      lines: number
    },
    linesCleared: number
  }
) {
  const state = args.afterState || args.beforeState;
  await ctx.db.insert("gameHistory", {
    playerId: args.playerId,
    gameId: args.gameId,
    sequence: Date.now(),
    action: args.action,
    pieceType: args.beforeState.currentPiece,
    position: args.beforeState.position,
    rotation: args.beforeState.rotation,
    linesCleared: args.linesCleared,
    board: args.beforeState.board,
    nextPiece: state.nextPiece,
    holdPiece: state.holdPiece,
    score: state.score,
    level: state.level
  });
}

// 점수 계산 함수 (하나만 유지)
function calculateScore(lines: number, level: number): number {
  const basePoints: Record<number, number> = {
    1: 100,
    2: 300,
    3: 500,
    4: 800
  };
  
  return (basePoints[lines] || 0) * Math.max(1, level);
}

// 모든 게임 액션을 처리하는 통합 함수
export const handleGameAction = mutation({
  args: {
    gameId: v.id("games"),
    playerId: v.id("players"),
    action: convexValidators.DIRECTIONS
  },
  handler: async (ctx, args) => {
    console.log('Action received:', args.action);
    
    const player = await ctx.db.get(args.playerId);
    if (!player) throw new Error("Player not found");
    
    // 액션 실행 전 상태 저장 (타입 호환성 보장)
    const beforeState = {
      board: player.board,
      currentPiece: player.currentPiece,
      position: player.position,
      rotation: player.rotation,
      nextPiece: player.nextPiece,
      holdPiece: player.holdPiece,
      score: player.score,
      level: player.level,
      lines: player.lines
    };

    // hold 액션 처리
    if (args.action === 'hold') {
      const currentPiece = player.currentPiece;
      const holdPiece = player.holdPiece;
      const nextPiece = player.nextPiece;

      const updates = {
        currentPiece: holdPiece || nextPiece,
        nextPiece: holdPiece ? getRandomPiece() : nextPiece,
        holdPiece: currentPiece,
        position: { x: 4, y: 0 },
        rotation: 0
      };

      await ctx.db.patch(args.playerId, updates);
      
      // 히스토리 저장
      if (player.gameId) {
        await _saveGameHistory(ctx, {
          gameId: player.gameId,
          playerId: args.playerId,
          action: args.action,
          beforeState,
          afterState: {
            ...beforeState,
            ...updates
          },
          linesCleared: 0
        });
      }

      return { clearedLines: 0, success: true };
    }

    const board = stringToBoard(player.board);
    const newPosition = { ...player.position };
    let newRotation = player.rotation;

    // 나머지 액션 처리
    switch (args.action) {
      case 'left':
        newPosition.x -= 1;
        break;
      case 'right':
        newPosition.x += 1;
        break;
      case 'down':
        newPosition.y += 1;
        break;
      case 'rotate':
        newRotation = (newRotation + 1) % 4;
        break;
      case 'hardDrop':
        while (!checkCollision(
          board,
          player.currentPiece,
          newPosition.x,
          newPosition.y + 1,
          newRotation
        )) {
          newPosition.y += 1;
        }
        // hardDrop 후 바로 피스 배치
        if (player.gameId) {
          await _saveGameHistory(ctx, {
            gameId: player.gameId,
            playerId: args.playerId,
            action: args.action,
            beforeState,
            linesCleared: 0
          });
        }
        await placePiece(ctx, args.playerId, {
          ...player,
          position: newPosition,
          rotation: newRotation,
          gameId: args.gameId
        });
        return { clearedLines: 0, success: true };
    }

    // 충돌 체크 (hardDrop은 이미 처리되었으므로 down만 체크)
    if (checkCollision(board, player.currentPiece, newPosition.x, newPosition.y, newRotation)) {
      if (args.action === 'down') {
        // 히스토리 저장
        if (player.gameId) {
          await _saveGameHistory(ctx, {
            gameId: player.gameId,
            playerId: args.playerId,
            action: args.action,
            beforeState,
            linesCleared: 0
          });
        }
        
        // 피스 배치
        await placePiece(ctx, args.playerId, {
          ...player,
          gameId: args.gameId
        });
        return { clearedLines: 0, success: true };
      }
      return { clearedLines: 0, success: false };
    }

    // 상태 업데이트
    const updates = {
      position: newPosition,
      rotation: newRotation
    };
    
    await ctx.db.patch(args.playerId, updates);
    
    // 히스토리 저장
    if (player.gameId) {
      await _saveGameHistory(ctx, {
        gameId: player.gameId,
        playerId: args.playerId,
        action: args.action,
        beforeState,
        afterState: {
          ...beforeState,
          ...updates
        },
        linesCleared: 0
      });
    }

    return { clearedLines: 0, success: true };
  }
});

// placePiece 함수 수정
export async function placePiece(
  ctx: MutationCtx, 
  playerId: Id<"players">, 
  player: DbPlayerState
) {
  // 피스 배치 전 상태 저장
  const beforeState = {
    board: player.board,
    currentPiece: player.currentPiece,
    position: player.position,
    rotation: player.rotation,
    nextPiece: player.nextPiece,
    holdPiece: player.holdPiece,
    score: player.score,
    level: player.level,
    lines: player.lines
  };

  const board = stringToBoard(player.board);
  const newBoard = board.map(row => [...row]);
  const pieceMatrix = PIECE_ROTATIONS[player.currentPiece][player.rotation];

  // 피스 배치
  for (let py = 0; py < pieceMatrix.length; py++) {
    for (let px = 0; px < pieceMatrix[py].length; px++) {
      if (pieceMatrix[py][px]) {
        const boardY = player.position.y + py;
        const boardX = player.position.x + px;
        if (boardY >= 0 && boardY < GAME_VALUES.INITIAL_BOARD_SIZE.HEIGHT && 
            boardX >= 0 && boardX < GAME_VALUES.INITIAL_BOARD_SIZE.WIDTH) {
          newBoard[boardY][boardX] = 1;
        }
      }
    }
  }

  const { updatedBoard: clearedBoard, linesCleared } = clearLines(newBoard);
  const newScore = player.score + calculateScore(linesCleared, player.level);
  const newLines = player.lines + linesCleared;
  const newLevel = Math.floor(newLines / 10) + 1;

  const nextPosition = { x: 4, y: 0 };
  const isGameOver = checkCollision(
    clearedBoard, 
    player.nextPiece,
    nextPosition.x,
    nextPosition.y,
    0
  );

  if (isGameOver) {
    if (!player.gameId) {
      throw new Error("Game ID not found");
    }
    await endGame(ctx, { 
      gameId: player.gameId, 
      playerId 
    });
    return;
  }

  const updates = {
    board: boardToString(clearedBoard),
    score: newScore,
    level: newLevel,
    lines: newLines,
    currentPiece: player.nextPiece,
    nextPiece: getRandomPiece(),
    position: nextPosition,
    rotation: 0
  };

  await ctx.db.patch(playerId, updates);

  // 피스 배치 후 상태로 히스토리 저장
  const updatedPlayer = await ctx.db.get(playerId) as DbPlayerState;
  if (updatedPlayer && player.gameId) {
    await _saveGameHistory(ctx, {
      gameId: player.gameId,
      playerId,
      action: 'down',  // 피스 배치는 'down' 액션으로 기록
      beforeState: {
        ...beforeState,
        holdPiece: undefined
      },
      afterState: {
        ...beforeState,
        ...updates,
        holdPiece: undefined
      },
      linesCleared
    });
  }
}

function stringToBoard(boardString: string): number[][] {
  const board = [];
  for (let i = 0; i < GAME_VALUES.INITIAL_BOARD_SIZE.HEIGHT; i++) {
    board.push(boardString.slice(i * GAME_VALUES.INITIAL_BOARD_SIZE.WIDTH, (i + 1) * GAME_VALUES.INITIAL_BOARD_SIZE.WIDTH).split('').map(Number));
  }
  return board;
}

function boardToString(board: number[][]): string {
  return board.flat().join('');
}

function checkCollision(
  board: number[][], 
  piece: tsValidators["PIECES"],
  x: number, 
  y: number,
  rotation: number
): boolean {
  const pieceMatrix = PIECE_ROTATIONS[piece][rotation];
  
  for (let py = 0; py < pieceMatrix.length; py++) {
    for (let px = 0; px < pieceMatrix[py].length; px++) {
      if (pieceMatrix[py][px]) {
        const boardX = x + px;
        const boardY = y + py;
        // 경계 체크
        if (boardX < 0 || boardX >= GAME_VALUES.INITIAL_BOARD_SIZE.WIDTH || boardY >= GAME_VALUES.INITIAL_BOARD_SIZE.HEIGHT) {
          return true;
        }

        if (boardY < 0) {
          continue;
        }

        if (board[boardY][boardX]) {
          return true;
        }
      }
    }
  }

  return false;
}

function clearLines(board: number[][]): { updatedBoard: number[][], linesCleared: number } {
  const newBoard = board.filter(row => !row.every(cell => cell === 1));
  const linesCleared = GAME_VALUES.INITIAL_BOARD_SIZE.HEIGHT - newBoard.length;
  
  for (let i = 0; i < linesCleared; i++) {
    newBoard.unshift(Array(GAME_VALUES.INITIAL_BOARD_SIZE.WIDTH).fill(0));
  }
  
  return {
    updatedBoard: newBoard,
    linesCleared
  };
}

// 게임 정지
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

// Ready 상태 정
export const setReady = mutation({
  args: {
    gameId: v.id("games"),
    playerId: v.id("players")
  },
  handler: async (ctx, args) => {
    // 1. 플레이어 존재 확인
    const player = await ctx.db.get(args.playerId);
    if (!player) {
      throw new Error("Player not found");
    }

    // 2. 게임 존재 확인
    const game = await ctx.db.get(args.gameId);
    if (!game) {
      throw new Error("Game not found");
    }

    // 3. 플레이어 상태 업데이트
    await ctx.db.patch(args.playerId, {
      gameId: args.gameId,  // 게임 ID 연결
      isReady: true,
      isPlaying: true
    });

    // 4. 모든 플레이어가 ready인지 확인
    const allPlayers = await Promise.all(
      game.players.map(pid => ctx.db.get(pid))
    );

    const allReady = allPlayers.every(p => p?.isReady);
    if (allReady) {
      await ctx.db.patch(args.gameId, { 
        status: "playing" 
      });
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

export const getGameState = query({
  args: { gameId: v.id("games") },
  handler: async (ctx, args) => {
    const game = await ctx.db.get(args.gameId);
    if (!game) return null;

    const players = await ctx.db
      .query("players")
      .filter(q => q.eq(q.field("gameId"), args.gameId))
      .collect();

    return {
      status: game.status,
      players: players.map(player => ({
        id: player._id,
        name: player.playerName,
        board: player.board,
        score: player.score,
        lines: player.lines,
        isPlaying: player.isPlaying
      })),
      currentPlayer: players.find(p => 
        p._id === localStorage.getItem('playerId')
      )
    };
  }
});

export const endGame = mutation({
  args: {
    gameId: v.id("games"),
    playerId: v.id("players")
  },
  handler: async (ctx, args) => {
    try {
      const player = await ctx.db.get(args.playerId);
      if (!player) throw new Error("Player not found");

      const gameStartTime = Date.now() - 1000 * 60 * 5; // 임시로 5분 전으로 설정
      
      // 게임 결과 저장
      await ctx.db.insert("gameResults", {
        gameId: args.gameId,
        playerId: args.playerId,
        finalScore: player.score,
        totalLines: player.lines,
        maxCombo: 0, // TODO: 콤보 시스템 구현 시 추가
        totalPieces: 0, // TODO: 피스 카운트 시스템 구현 시 추가
        startTime: gameStartTime,
        endTime: Date.now(),
        averageSpeed: 0 // TODO: 평균 의사결정 시간 구현 시 추가
      });

      // 1. 게임 상태를 finished 변경
      await ctx.db.patch(args.gameId, {
        status: "finished",
        winnerId: args.playerId
      });

      // 2. 모든 플레이어 상태 초기화
      const players = await ctx.db
        .query("players")
        .filter(q => q.eq(q.field("gameId"), args.gameId))
        .collect();

      for (const player of players) {
        await ctx.db.patch(player._id, {
          gameId: undefined,
          isPlaying: false,
          isReady: false,
          score: 0,
          level: 1,
          lines: 0,
          board: "0".repeat(GAME_VALUES.INITIAL_BOARD_SIZE.WIDTH * GAME_VALUES.INITIAL_BOARD_SIZE.HEIGHT),
          currentPiece: getRandomPiece(),
          nextPiece: getRandomPiece(),
          holdPiece: undefined,
          position: { x: 4, y: 0 },
          rotation: 0,
          garbageLines: 0
        });
      }

      return { success: true };
    } catch (error) {
      console.error('Error in endGame mutation:', error);
      throw error;
    }
  }
});

// 방 라이어 보내기 기능
export const sendGarbageLines = mutation({
  args: {
    gameId: v.id("games"),
    fromPlayerId: v.id("players"),
    toPlayerId: v.id("players"),
    lineCount: v.number()
  },
  handler: async (ctx, args) => {
    const targetPlayer = await ctx.db.get(args.toPlayerId);
    if (!targetPlayer) return;
    
    await ctx.db.patch(args.toPlayerId, {
      garbageLines: (targetPlayer.garbageLines || 0) + args.lineCount
    });
  }
});

export const syncState = mutation({
  args: {
    gameId: v.id("games"),
    state: v.object({
      board: v.string(),
      currentPiece: convexValidators.PIECES,
      nextPiece: convexValidators.PIECES,
      holdPiece: v.optional(convexValidators.PIECES),
      score: v.number(),
      level: v.number(),
      lines: v.number()
    })
  },
  handler: async (ctx, args) => {
    const { gameId, state } = args;
    const player = await ctx.db
      .query("players")
      .filter(q => q.eq(q.field("gameId"), gameId))
      .first();
    
    if (!player) return null;

    // 플레이어 상태 직접 업데이트
    await ctx.db.patch(player._id, {
      board: state.board,
      currentPiece: state.currentPiece,
      nextPiece: state.nextPiece,
      holdPiece: state.holdPiece,
      score: state.score,
      level: state.level,
      lines: state.lines
    });

    return state;
  }
});

// getPlayers 쿼리 추가
export const getPlayers = query({
  args: { gameId: v.id("games") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("players")
      .filter(q => q.eq(q.field("gameId"), args.gameId))
      .collect();
  }
});

// startGame mutation 추가
export const startGame = mutation({
  args: { gameId: v.id("games") },
  handler: async (ctx, args) => {
    const game = await ctx.db.get(args.gameId);
    if (!game) return;

    await ctx.db.patch(args.gameId, {
      status: "playing"
    });
  }
});

// 새로운 mutation 추가
export const startGameAfterDelay = mutation({
  args: {
    gameId: v.id("games")
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.gameId, {
      status: "playing"
    });
  }
});

// AI 플레이어 생성
export const createAIPlayer = mutation({
  args: {
    gameId: v.id("games"),
    playerName: v.string()
  },
  handler: async (ctx, args) => {
    const aiPlayerId = await ctx.db.insert("players", {
      gameId: args.gameId,
      playerName: args.playerName,
      score: 0,
      level: 1,
      lines: 0,
      board: "0".repeat(GAME_VALUES.INITIAL_BOARD_SIZE.WIDTH * GAME_VALUES.INITIAL_BOARD_SIZE.HEIGHT),
      currentPiece: getRandomPiece(),
      nextPiece: getRandomPiece(),
      position: { x: 4, y: 0 },
      rotation: 0,
      isPlaying: false,
      isReady: false,
      garbageLines: 0,
      isAI: true
    });

    // 임에 AI 플레이어 추가
    const game = await ctx.db.get(args.gameId);
    if (game) {
      await ctx.db.patch(args.gameId, {
        players: [...game.players, aiPlayerId]
      });
    }

    return aiPlayerId;
  }
});

// 게임 히스토리 저장 mutation
export const saveGameHistory = mutation({
  args: {
    playerId: v.id("players"),
    gameId: v.id("games"),
    action: convexValidators.DIRECTIONS,
    pieceType: convexValidators.PIECES,
    position: v.object({
      x: v.number(),
      y: v.number()
    }),
    rotation: v.number(),
    linesCleared: v.number(),
    board: v.string(),
    nextPiece: convexValidators.PIECES,
    holdPiece: v.optional(convexValidators.PIECES),
    score: v.number(),
    level: v.number()
  },
  handler: async (ctx, args) => {
    // 현재 게임의 마지막 시퀀스 번호 조회
    const lastHistory = await ctx.db
      .query("gameHistory")
      .filter(q => q.eq(q.field("gameId"), args.gameId))
      .order("desc")
      .first();

    const sequence = lastHistory ? (lastHistory.sequence + 1) : 0;

    // 새로운 히스토리 저장
    return await ctx.db.insert("gameHistory", {
      playerId: args.playerId,
      gameId: args.gameId,
      sequence,
      action: args.action,
      pieceType: args.pieceType,
      position: args.position,
      rotation: args.rotation,
      linesCleared: args.linesCleared,
      board: args.board,
      nextPiece: args.nextPiece,
      holdPiece: args.holdPiece,
      score: args.score,
      level: args.level
    });
  }
});

// 게임 결과 저장 mutation
export const saveGameResult = mutation({
  args: {
    gameId: v.id("games"),
    playerId: v.id("players"),
    finalScore: v.number(),
    totalLines: v.number(),
    maxCombo: v.number(),
    totalPieces: v.number(),
    startTime: v.number(),
    endTime: v.number(),
    averageSpeed: v.number()
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("gameResults", {
      ...args
    });
  }
});

// 게임 히스토리 조회 query
export const getGameHistory = query({
  args: { 
    gameId: v.id("games"),
    limit: v.optional(v.number())
  },
  handler: async (ctx, args) => {
    const query = ctx.db
      .query("gameHistory")
      .filter(q => q.eq(q.field("gameId"), args.gameId))
      .order("asc");
    
    const results = args.limit 
      ? await query.take(args.limit)
      : await query.collect();

    return results;
  }
});

// 플레이어의 게임 히스토리 조회 query
export const getPlayerHistory = query({
  args: { 
    playerId: v.id("players"),
    limit: v.optional(v.number())
  },
  handler: async (ctx, args) => {
    const query = ctx.db
      .query("gameHistory")
      .filter(q => q.eq(q.field("playerId"), args.playerId))
      .order("desc");
    
    const results = args.limit 
      ? await query.take(args.limit)
      : await query.collect();

    return results;
  }
});

// 게임 결과 조회 query
export const getGameResult = query({
  args: { gameId: v.id("games") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("gameResults")
      .filter(q => q.eq(q.field("gameId"), args.gameId))
      .first();
  }
});

// 점수 업데이트 함수
export const updateScore = mutation({
  args: {
    gameId: v.id('games'),
    playerId: v.id('players'),
    score: v.number(),
    level: v.number(),
    lines: v.number()
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.playerId, {
      score: args.score,
      level: args.level,
      lines: args.lines
    });
  }
});

