# QueryGPT Evaluation Frontend

A modern web interface for evaluating and interacting with the QueryGPT NL-to-SQL system.

## Features

- 💬 **Chat Interface**: Natural language to SQL queries with real-time results
- 📊 **Profiling Viewer**: View detailed evaluation results from test runs
- 📋 **Golden Dataset Viewer**: Browse and search the evaluation dataset
- 📈 **Metrics Dashboard**: Visualize accuracy, latency, and cost metrics

## Setup

### Backend Setup

1. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Make sure your `.env` file is in the project root with:
```
FIREWORKS_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
FIREWORKS_MODEL=fireworks/kimi-k2-thinking
FIREWORKS_INTENT_MODEL=fireworks/kimi-k2-thinking
```

3. Start the backend server:
```bash
cd backend
python main.py
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## Usage

1. **Chat Interface** (`/`): 
   - Select Fireworks or OpenAI provider
   - Type natural language questions
   - View generated SQL and results
   - See performance metrics

2. **Profiling Viewer** (`/profiling`):
   - View evaluation results from `evaluation_results_fireworks.json` and `evaluation_results_openai.json`
   - Compare providers side-by-side
   - Drill down into individual test cases

3. **Golden Dataset** (`/golden-dataset`):
   - Browse all test cases
   - Filter by category
   - Search questions and SQL
   - View expected results

4. **Metrics Dashboard** (`/metrics`):
   - Visualize accuracy metrics
   - Compare latency between providers
   - View token usage (cost proxy)
   - See summary statistics

## API Endpoints

The backend provides the following endpoints:

- `GET /api/health` - Health check
- `POST /api/chat` - Process NL-to-SQL query
- `GET /api/evaluation-results` - Get all evaluation results
- `GET /api/evaluation-results/{provider}` - Get results for specific provider
- `GET /api/golden-dataset` - Get golden dataset
- `GET /api/golden-dataset/stats` - Get dataset statistics

## Development

### Backend
- Uses FastAPI for the API server
- Integrates with existing `fireworks_querygpt.py` and `evaluate_agents.py`
- CORS enabled for frontend development

### Frontend
- React 18 with Vite
- React Router for navigation
- Recharts for data visualization
- Axios for API calls

## Troubleshooting

1. **Backend won't start**: 
   - Check that all dependencies are installed
   - Verify `.env` file exists with API keys
   - Ensure evaluation result files exist in project root

2. **Frontend can't connect to backend**:
   - Verify backend is running on port 8000
   - Check CORS settings in `backend/main.py`
   - Verify proxy settings in `frontend/vite.config.js`

3. **No evaluation results shown**:
   - Run `python3 run_full_evaluation.py` first to generate results
   - Check that result files are in the project root directory

