import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

/**
 * 테트리스 피스 타입 정의
 * I: 직선 모양, O: 정사각형 모양, T: T자 모양
 * S, Z: S/Z 모양, J, L: J/L자 모양
 */
const TETROMINOS = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
const getRandomPiece = () => TETROMINOS[Math.floor(Math.random() * TETROMINOS.length)];

/**
 * 게임 테이블 스키마
 * @property status - 게임 상태 ("waiting" | "playing" | "finished")
 * @property players - 참가자 ID 배열
 * @property winnerId - 승자 ID (게임 종료 시)
 */

/**
 * 플레이어 테이블 스키마
 * @property gameId - 참여 중인 게임 ID
 * @property name - 플레이어 이름
 * @property score - 현재 점수
 * @property board - 게임판 상태 (20x10 문자열)
 * @property currentPiece - 현재 조작 중인 피스
 * @property nextPiece - 다음 피스
 * @property isPlaying - 게임 진행 중 여부
 */

/**
 * 새로운 게임을 생성합니다.
 * 첫 플레이어를 생성하고 게임에 등록합니다.
 */
export const createGame = mutation({...});

/**
 * 기존 게임에 참가합니다.
 * 대기 중인 게임에만 참가할 수 있습니다.
 */
export const joinGame = mutation({...});

/**
 * 게임을 시작합니다.
 * 최소 2명의 플레이어가 필요합니다.
 */
export const startGame = mutation({...});

/**
 * 플레이어의 게임 상태를 업데이트합니다.
 * 보드 상태, 점수, 현재 피스가 업데이트되며
 * 새로운 다음 피스가 할당됩니다.
 */
export const updatePlayer = mutation({...});

/**
 * 게임을 종료하고 승자를 기록합니다.
 */
export const endGame = mutation({...});

/**
 * 진행 중인 게임 목록을 조회합니다.
 * 종료된 게임은 제외됩니다.
 */
export const listGames = query({...});