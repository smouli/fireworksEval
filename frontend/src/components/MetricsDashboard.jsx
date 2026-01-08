import React, { useState, useEffect } from 'react'
import { api } from '../utils/api'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import './MetricsDashboard.css'

const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe']

function MetricsDashboard() {
  const [results, setResults] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = async () => {
    try {
      const response = await api.get('/api/evaluation-results')
      setResults(response.data)
    } catch (error) {
      console.error('Error loading results:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="card">
        <div className="loading">Loading metrics...</div>
      </div>
    )
  }

  const providers = Object.keys(results)
  if (providers.length === 0) {
    return (
      <div className="card">
        <div className="error">No evaluation results found.</div>
      </div>
    )
  }

  // Format provider name for display
  const formatProviderName = (provider) => {
    if (provider === 'fireworks_finetuned') {
      return 'Fireworks Finetuned'
    }
    return provider.charAt(0).toUpperCase() + provider.slice(1)
  }

  // Prepare accuracy data
  const accuracyData = providers.map((provider) => {
    const summary = results[provider]?.summary || {}
    return {
      provider: formatProviderName(provider),
      overall: (summary.overall?.accuracy || 0) * 100,
      intent: (summary.intent_agent?.accuracy || 0) * 100,
      table: (summary.table_agent?.avg_f1 || 0) * 100,
      sql_similarity: (summary.sql_generation_agent?.avg_sql_similarity || 0) * 100,
      result_match: (summary.sql_generation_agent?.result_match_rate || 0) * 100,
    }
  })

  // Prepare latency data
  const latencyData = providers.map((provider) => {
    const latency = results[provider]?.summary?.latency_metrics || {}
    return {
      provider: formatProviderName(provider),
      intent: latency.intent_agent?.avg_latency_ms || 0,
      table: latency.table_agent?.avg_latency_ms || 0,
      column_prune: latency.column_prune_agent?.avg_latency_ms || 0,
      sql_generation: latency.sql_generation_agent?.avg_latency_ms || 0,
    }
  })

  // Calculate total latency
  const totalLatencyData = latencyData.map((item) => ({
    provider: item.provider,
    total: item.intent + item.table + item.column_prune + item.sql_generation,
  }))

  // Prepare token usage data (cost proxy)
  const tokenData = providers.map((provider) => {
    const latency = results[provider]?.summary?.latency_metrics || {}
    return {
      provider: formatProviderName(provider),
      intent: latency.intent_agent?.total_tokens || 0,
      table: latency.table_agent?.total_tokens || 0,
      column_prune: latency.column_prune_agent?.total_tokens || 0,
      sql_generation: latency.sql_generation_agent?.total_tokens || 0,
    }
  })

  // Calculate total tokens
  const totalTokenData = tokenData.map((item) => ({
    provider: item.provider,
    total: item.intent + item.table + item.column_prune + item.sql_generation,
  }))

  // Agent accuracy breakdown
  const agentAccuracyData = providers.flatMap((provider) => {
    const summary = results[provider]?.summary || {}
    return [
      {
        agent: 'Intent',
        [provider]: (summary.intent_agent?.accuracy || 0) * 100,
      },
      {
        agent: 'Table',
        [provider]: (summary.table_agent?.avg_f1 || 0) * 100,
      },
      {
        agent: 'SQL Similarity',
        [provider]: (summary.sql_generation_agent?.avg_sql_similarity || 0) * 100,
      },
    ]
  })

  return (
    <div className="metrics-dashboard">
      <div className="card">
        <div className="card-title">📈 Evaluation Metrics Dashboard</div>
      </div>

      <div className="metrics-grid">
        {/* Accuracy Comparison */}
        <div className="card chart-card">
          <div className="chart-title">Accuracy Comparison</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={accuracyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="provider" />
              <YAxis domain={[0, 100]} />
              <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="overall" fill="#667eea" name="Overall" />
              <Bar dataKey="intent" fill="#764ba2" name="Intent" />
              <Bar dataKey="table" fill="#f093fb" name="Table (F1)" />
              <Bar dataKey="sql_similarity" fill="#4facfe" name="SQL Similarity" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Latency Comparison */}
        <div className="card chart-card">
          <div className="chart-title">Latency Comparison (ms)</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={latencyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="provider" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="intent" fill="#667eea" name="Intent Agent" />
              <Bar dataKey="table" fill="#764ba2" name="Table Agent" />
              <Bar dataKey="column_prune" fill="#f093fb" name="Column Prune" />
              <Bar dataKey="sql_generation" fill="#4facfe" name="SQL Generation" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Total Latency */}
        <div className="card chart-card">
          <div className="chart-title">Total Pipeline Latency</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={totalLatencyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="provider" />
              <YAxis />
              <Tooltip formatter={(value) => `${value.toFixed(0)}ms`} />
              <Bar dataKey="total" fill="#667eea" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Token Usage (Cost Proxy) */}
        <div className="card chart-card">
          <div className="chart-title">Token Usage (Cost Proxy)</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={tokenData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="provider" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="intent" fill="#667eea" name="Intent Agent" />
              <Bar dataKey="table" fill="#764ba2" name="Table Agent" />
              <Bar dataKey="column_prune" fill="#f093fb" name="Column Prune" />
              <Bar dataKey="sql_generation" fill="#4facfe" name="SQL Generation" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Total Tokens */}
        <div className="card chart-card">
          <div className="chart-title">Total Token Usage</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={totalTokenData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="provider" />
              <YAxis />
              <Tooltip formatter={(value) => value.toLocaleString()} />
              <Bar dataKey="total" fill="#764ba2" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Result Match Rate */}
        <div className="card chart-card">
          <div className="chart-title">Result Match Rate</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={accuracyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="provider" />
              <YAxis domain={[0, 100]} />
              <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
              <Bar dataKey="result_match" fill="#4facfe" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="card">
        <div className="card-title">Summary Statistics</div>
        <div className="stats-grid">
          {providers.map((provider) => {
            const summary = results[provider]?.summary || {}
            const latency = summary.latency_metrics || {}
            return (
              <div key={provider} className="provider-stats">
                <h3>{formatProviderName(provider)}</h3>
                <div className="stat-row">
                  <span className="stat-label">Overall Accuracy:</span>
                  <span className="stat-value">
                    {((summary.overall?.accuracy || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Total Latency:</span>
                  <span className="stat-value">
                    {Object.values(latency).reduce(
                      (sum, m) => sum + (m.avg_latency_ms || 0),
                      0
                    ).toFixed(0)}ms
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Total Tokens:</span>
                  <span className="stat-value">
                    {Object.values(latency).reduce(
                      (sum, m) => sum + (m.total_tokens || 0),
                      0
                    ).toLocaleString()}
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Result Match:</span>
                  <span className="stat-value">
                    {((summary.sql_generation_agent?.result_match_rate || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default MetricsDashboard

