#!/bin/bash
# Start script for QueryGPT Frontend

echo "🚀 Starting QueryGPT Frontend..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv_querygpt" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv_querygpt/bin/activate

# Check if backend dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing backend dependencies..."
    pip install -r backend/requirements.txt
fi

# Start backend in background
echo "🔧 Starting backend server..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Check if backend is running
if ! curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "⚠️  Backend may not have started correctly. Check the output above."
else
    echo "✅ Backend is running on http://localhost:8000"
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start frontend
echo "🎨 Starting frontend server..."
echo ""
echo "Frontend will be available at http://localhost:5173"
echo "Backend API is available at http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

cd frontend
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT

