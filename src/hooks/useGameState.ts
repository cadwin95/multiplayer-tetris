import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { Id } from "../../convex/_generated/dataModel";

export function useGameState(gameId: Id<"games">) {
  const game = useQuery(api.games.getGame, { gameId });
  const players = useQuery(api.games.getPlayers, { gameId });
  const currentPlayer = useQuery(api.games.getPlayer, { 
    playerId: localStorage.getItem('playerId') as Id<"players"> 
  });

  const moveTetrominoe = useMutation(api.games.moveTetrominoe);
  const rotateTetrominoe = useMutation(api.games.rotateTetrominoe);
  const hardDrop = useMutation(api.games.hardDrop);
  const pauseGame = useMutation(api.games.pauseGame);

  console.log('useGameState data:', {
    game,
    players,
    subscriptionStatus: game?._subscription?.status,
  });

  return {
    game,
    players,
    currentPlayer,
    moveTetrominoe,
    rotateTetrominoe,
    hardDrop,
    pauseGame
  };
}