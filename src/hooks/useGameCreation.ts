import { useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { Id } from "../../convex/_generated/dataModel";
import { useNavigate } from "react-router-dom";

type GameMode = "solo" | "ai" | "multi";

export function useGameCreation() {
  const navigate = useNavigate();
  const createGame = useMutation(api.games.createGame);
  const createAIPlayer = useMutation(api.games.createAIPlayer);
  const setReady = useMutation(api.games.setReady);

  const startGame = async (mode: GameMode) => {
    const playerId = localStorage.getItem('playerId') as Id<"players">;
    
    try {
      const { gameId } = await createGame({ playerId, mode });

      if (mode === "ai") {
        const aiPlayerId = await createAIPlayer({
          gameId,
          playerName: "AI Player"
        });
        await setReady({ gameId, playerId: aiPlayerId });
      }

      await setReady({ gameId, playerId });
      navigate(`/game/${gameId}`);
    } catch (error) {
      console.error('Failed to create game:', error);
      navigate('/');
    }
  };

  return { startGame };
} 