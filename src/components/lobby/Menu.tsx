import { Link } from 'react-router-dom';

export function Menu() {
  return (
    <div className="flex flex-col gap-4">
      <Link 
        to="/support" 
        className="menu-item bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-500 hover:to-orange-600"
      >
        <h2>SUPPORT TETR.IO</h2>
        <p className="text-sm opacity-80">SUPPORT DEVELOPMENT OR GIFT SUPPORTER STATUS AND GET REWARDS!</p>
      </Link>

      <Link 
        to="/store" 
        className="menu-item bg-gradient-to-r from-red-800 to-red-900 hover:from-red-700 hover:to-red-800"
      >
        <h2>MERCH STORE</h2>
        <p className="text-sm opacity-80">BROWSE THE COLLECTION OF ALL OFFICIAL TETRIS GOODIES!</p>
      </Link>

      <Link 
        to="/download" 
        className="menu-item bg-gradient-to-r from-gray-700 to-gray-800 hover:from-gray-600 hover:to-gray-700"
      >
        <h2>TETRIS DESKTOP</h2>
        <p className="text-sm opacity-80">GET THE DESKTOP CLIENT FOR EXTRA FUNCTIONS AND IMPROVED PERFORMANCE</p>
      </Link>

      <Link 
        to="/patch-notes" 
        className="menu-item bg-gradient-to-r from-gray-800 to-gray-900 hover:from-gray-700 hover:to-gray-800"
      >
        <h2>PATCH NOTES</h2>
        <p className="text-sm opacity-80">FOLLOW DEVELOPMENT AND SEE WHAT'S NEW</p>
      </Link>
    </div>
  );
} 