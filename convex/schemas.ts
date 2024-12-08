// schema.ts
import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

const GameStatus = v.union(
  v.literal("waiting"),
  v.literal("playing"),
  v.literal("paused"),
  v.literal("finished")
);

export default defineSchema({
  games: defineTable({
    status: GameStatus,
    players: v.array(v.id("players")),
    settings: v.object({
      startLevel: v.number(),
      gameSpeed: v.number(),
    }),
    winnerId: v.optional(v.id('players')),
  }),

  players: defineTable({
    gameId: v.id('games'),
    name: v.string(),
    score: v.number(),
    level: v.number(),
    lines: v.number(),
    board: v.string(),
    currentPiece: v.string(),
    nextPiece: v.string(),
    position: v.object({
      x: v.number(),
      y: v.number(),
    }),
    isReady: v.boolean(),
    isPlaying: v.boolean(),
    garbageLines: v.number(), // 받은 방해 라인 수
  }).index('by_game', ['gameId']),
});