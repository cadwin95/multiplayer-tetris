import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { Id } from "../../convex/_generated/dataModel";
import { LocalGameState } from '../types/game';

export function useGameState(gameId: Id<"games">) {
  // 서버 상태
  const game = useQuery(api.games.getGame, { gameId });
  const players = useQuery(api.games.getPlayers, { gameId });
  const currentPlayer = useQuery(api.games.getPlayer, { 
    playerId: localStorage.getItem('playerId') as Id<"players"> 
  });

  // 로컬 상태
  const [localState, setLocalState] = useState<LocalGameState | null>(null);
  
  // 서버 뮤테이션
  const moveTetrominoe = useMutation(api.games.moveTetrominoe);
  const rotateTetrominoe = useMutation(api.games.rotateTetrominoe);
  const hardDrop = useMutation(api.games.hardDrop);
  const pauseGame = useMutation(api.games.pauseGame);

  // 서버 상태로 로컬 상태 초기화
  useEffect(() => {
    if (currentPlayer?.gameState) {
      setLocalState(currentPlayer.gameState);
    }
  }, [currentPlayer?.gameState]);

  // 로컬 이동 처리
  const handleMove = useCallback(async (direction: 'left' | 'right' | 'down') => {
    if (!localState) return;

    // 로컬 상태 즉시 업데이트
    const newPosition = {
      x: localState.position.x + (direction === 'left' ? -1 : direction === 'right' ? 1 : 0),
      y: localState.position.y + (direction === 'down' ? 1 : 0)
    };

    setLocalState(prev => ({
      ...prev!,
      position: newPosition,
      isValid: undefined
    }));

    // 서버 검증
    try {
      await moveTetrominoe({
        gameId,
        playerId: localStorage.getItem('playerId') as Id<"players">,
        direction
      });
    } catch (error) {
      // 서버 검증 실패시 원래 상태로 복구
      setLocalState(currentPlayer?.gameState || null);
    }
  }, [localState, gameId, moveTetrominoe, currentPlayer?.gameState]);

  // 로컬 회전 처리
  const handleRotate = useCallback(async () => {
    if (!localState) return;

    // 로컬 회전 즉시 적용
    const rotatedPiece = rotateMatrix(localState.currentPiece);
    setLocalState(prev => ({
      ...prev!,
      currentPiece: rotatedPiece,
      isValid: undefined
    }));

    // 서버 검증
    try {
      await rotateTetrominoe({
        gameId,
        playerId: localStorage.getItem('playerId') as Id<"players">
      });
    } catch (error) {
      setLocalState(currentPlayer?.gameState || null);
    }
  }, [localState, gameId, rotateTetrominoe, currentPlayer?.gameState]);``

  return {
    game,
    players,
    currentPlayer,
    gameState: localState || currentPlayer?.gameState,
    moveTetrominoe: handleMove,
    rotateTetrominoe: handleRotate,
    hardDrop,
    pauseGame
  };
}

// 회전 행렬 헬퍼 함수
function rotateMatrix(piece: string): string {
  // 피스 회전 로직 구현
  return piece;
}