import React, { useState, useEffect } from 'react'
import { api } from '../utils/api'
import './ProfilingViewer.css'

function ProfilingViewer() {
  const [results, setResults] = useState({})
  const [loading, setLoading] = useState(true)
  const [selectedProvider, setSelectedProvider] = useState(null)
  const [selectedTestCase, setSelectedTestCase] = useState(null)

  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = async () => {
    try {
      const response = await api.get('/api/evaluation-results')
      setResults(response.data)
      if (Object.keys(response.data).length > 0) {
        setSelectedProvider(Object.keys(response.data)[0])
      }
    } catch (error) {
      console.error('Error loading results:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="card">
        <div className="loading">Loading evaluation results...</div>
      </div>
    )
  }

  if (Object.keys(results).length === 0) {
    return (
      <div className="card">
        <div className="error">
          No evaluation results found. Run an evaluation first.
        </div>
      </div>
    )
  }

  const providers = Object.keys(results)
  const currentResults = selectedProvider ? results[selectedProvider] : null

  return (
    <div className="profiling-container">
      <div className="card">
        <div className="card-title">📊 Evaluation Results</div>
        <div className="provider-tabs">
          {providers.map((provider) => (
            <button
              key={provider}
              className={`provider-tab ${selectedProvider === provider ? 'active' : ''}`}
              onClick={() => {
                setSelectedProvider(provider)
                setSelectedTestCase(null)
              }}
            >
              {provider.charAt(0).toUpperCase() + provider.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {currentResults && (
        <>
          <div className="card">
            <div className="card-title">Summary</div>
            <div className="summary-grid">
              <div className="summary-item">
                <div className="summary-label">Overall Accuracy</div>
                <div className="summary-value">
                  {(currentResults.summary?.overall?.accuracy * 100 || 0).toFixed(1)}%
                </div>
                <div className="summary-detail">
                  {currentResults.summary?.overall?.correct || 0} / {currentResults.summary?.overall?.total || 0}
                </div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Intent Agent</div>
                <div className="summary-value">
                  {(currentResults.summary?.intent_agent?.accuracy * 100 || 0).toFixed(1)}%
                </div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Table Agent (F1)</div>
                <div className="summary-value">
                  {(currentResults.summary?.table_agent?.avg_f1 * 100 || 0).toFixed(1)}%
                </div>
              </div>
              <div className="summary-item">
                <div className="summary-label">SQL Similarity</div>
                <div className="summary-value">
                  {(currentResults.summary?.sql_generation_agent?.avg_sql_similarity * 100 || 0).toFixed(1)}%
                </div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Result Match</div>
                <div className="summary-value">
                  {(currentResults.summary?.sql_generation_agent?.result_match_rate * 100 || 0).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Test Cases</div>
            <div className="test-cases-list">
              {currentResults.test_cases?.map((testCase, idx) => (
                <div
                  key={idx}
                  className={`test-case-item ${selectedTestCase === idx ? 'selected' : ''}`}
                  onClick={() => setSelectedTestCase(selectedTestCase === idx ? null : idx)}
                >
                  <div className="test-case-header">
                    <span className="test-case-number">#{idx + 1}</span>
                    <span className="test-case-question">{testCase.question}</span>
                    <span className={`test-case-status ${testCase.correct ? 'correct' : 'incorrect'}`}>
                      {testCase.correct ? '✓' : '✗'}
                    </span>
                  </div>
                  {selectedTestCase === idx && (
                    <div className="test-case-details">
                      <div className="detail-section">
                        <div className="detail-label">Expected SQL:</div>
                        <pre className="detail-sql">{testCase.expected_sql}</pre>
                      </div>
                      <div className="detail-section">
                        <div className="detail-label">Generated SQL:</div>
                        <pre className="detail-sql">{testCase.generated_sql || 'N/A'}</pre>
                      </div>
                      <div className="detail-section">
                        <div className="detail-label">SQL Similarity:</div>
                        <div className="detail-value">
                          {((testCase.sql_similarity || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="detail-section">
                        <div className="detail-label">Result Match:</div>
                        <div className={`detail-value ${testCase.result_match ? 'match' : 'no-match'}`}>
                          {testCase.result_match ? '✓ Match' : '✗ No Match'}
                        </div>
                      </div>
                      {testCase.agent_results && (
                        <div className="detail-section">
                          <div className="detail-label">Agent Results:</div>
                          <div className="agent-results">
                            <div className="agent-result-item">
                              <span>Intent:</span>
                              <span>{testCase.agent_results.intent || 'N/A'}</span>
                            </div>
                            <div className="agent-result-item">
                              <span>Tables:</span>
                              <span>{testCase.agent_results.tables?.join(', ') || 'N/A'}</span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default ProfilingViewer

