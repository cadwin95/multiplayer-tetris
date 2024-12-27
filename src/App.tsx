// src/App.tsx
import { MainNav } from './components/layout/MainNav';
import Routes from './Routes';

function App() {
  return (
    <div className="min-h-screen bg-gray-900">
      <MainNav />
      <main className="pt-32">
        <Routes />
      </main>
    </div>
  );
}

export default App;