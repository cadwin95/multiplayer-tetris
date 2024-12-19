import { useNavigate } from "react-router-dom";
import { useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { Id } from "../../convex/_generated/dataModel";
import { useEffect } from 'react';

export default function AIGame() {
  const navigate = useNavigate();
  const createGame = useMutation(api.games.createGame);
  const createAIPlayer = useMutation(api.games.createAIPlayer);
  const setReady = useMutation(api.games.setReady);

  useEffect(() => {
    const startAIGame = async () => {
      try {
        // 1. 게임 생성
        const { gameId } = await createGame({
          playerId: localStorage.getItem('playerId') as Id<"players">,
          mode: "ai"
        });

        // 2. AI 플레이어 생성 및 게임에 추가
        const aiPlayerId = await createAIPlayer({
          gameId,
          playerName: "AI Player"
        });

        // 3. 두 플레이어 모두 ready 상태로 설정
        await setReady({
          gameId,
          playerId: localStorage.getItem('playerId') as Id<"players">
        });

        await setReady({
          gameId,
          playerId: aiPlayerId
        });

        // 4. 게임 화면으로 이동
        navigate(`/game/${gameId}`);
      } catch (error) {
        console.error('Failed to create AI game:', error);
        navigate('/');
      }
    };

    startAIGame();
  }, [createGame, createAIPlayer, setReady, navigate]);

  return <div className="text-white text-center">Setting up AI game...</div>;
} 