import React from 'react';
import FileUploader from './components/FileUploader';

const App = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <header className="mb-4">
        <h1 className="text-3xl font-bold">PDF2MD</h1>
      </header>
      <main>
        <FileUploader />
      </main>
    </div>
  );
};

export default App;
