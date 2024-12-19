import { Link } from 'react-router-dom';

export function MainNav() {
  const nickname = localStorage.getItem('nickname') || 'ANONYMOUS';

  // 닉네임 입력 화면에서는 네비게이션 숨기기
  if (!localStorage.getItem('nickname')) {
    return null;
  }

  const handleLogout = () => {
    localStorage.removeItem('nickname');  // 닉네임 삭제
    localStorage.removeItem('playerId');  // 플레이어 ID도 삭제
    window.location.href = '/';  // 홈으로 리다이렉트
  };

  return (
    <div className="fixed w-full top-0 left-0 z-40">
      <div className="flex justify-between items-center px-6 py-3 bg-gray-900/80 backdrop-blur-sm">
        <Link to="/" className="text-[#00f0ff] text-2xl font-bold hover:text-[#00ff87] transition-colors">
          HOME
        </Link>
        <div className="flex items-center gap-4">
          <div className="text-white font-mono px-4 py-2 bg-red-500/20 rounded">
            {nickname}
          </div>
          <button 
            onClick={handleLogout}
            className="text-red-400 hover:text-red-300 px-3 py-1 rounded border border-red-500/30 hover:border-red-500/50"
          >
            Logout
          </button>
        </div>
      </div>
    </div>
  );
} 