import React, { useState, useEffect } from 'react'
import { api } from '../utils/api'
import './GoldenDatasetViewer.css'

function GoldenDatasetViewer() {
  const [dataset, setDataset] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedItem, setSelectedItem] = useState(null)
  const [filterCategory, setFilterCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    loadDataset()
    loadStats()
  }, [])

  const loadDataset = async () => {
    try {
      setError(null)
      const response = await api.get('/api/golden-dataset')
      setDataset(response.data || [])
    } catch (error) {
      console.error('Error loading dataset:', error)
      setError(error.message || 'Failed to load golden dataset. Check console for details.')
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const response = await api.get('/api/golden-dataset/stats')
      setStats(response.data)
    } catch (error) {
      console.error('Error loading stats:', error)
    }
  }

  const categories = stats ? Object.keys(stats.categories || {}) : []
  const filteredDataset = dataset.filter((item) => {
    const matchesCategory = filterCategory === 'all' || item.category === filterCategory
    const matchesSearch = searchQuery === '' || 
      item.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ((item.sql || item.expected_sql) && (item.sql || item.expected_sql).toLowerCase().includes(searchQuery.toLowerCase()))
    return matchesCategory && matchesSearch
  })

  if (loading) {
    return (
      <div className="card">
        <div className="loading">Loading golden dataset...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card">
        <div className="error">Error: {error}</div>
      </div>
    )
  }

  return (
    <div className="golden-dataset-container">
      <div className="card">
        <div className="card-title">📋 Golden Dataset</div>
        {stats && (
          <div className="dataset-stats">
            <div className="stat-item">
              <span className="stat-label">Total Cases:</span>
              <span className="stat-value">{stats.total_cases}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Categories:</span>
              <span className="stat-value">{stats.categories_count}</span>
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <div className="filters-container">
          <div className="filter-group">
            <label>Category:</label>
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="filter-select"
            >
              <option value="all">All Categories</option>
              {categories.map((cat) => (
                <option key={cat} value={cat}>
                  {cat} ({stats?.categories[cat] || 0})
                </option>
              ))}
            </select>
          </div>
          <div className="filter-group">
            <label>Search:</label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search questions or SQL..."
              className="filter-input"
            />
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          Dataset Items ({filteredDataset.length} {filteredDataset.length === 1 ? 'item' : 'items'})
        </div>
        <div className="dataset-list">
          {filteredDataset.map((item, idx) => (
            <div
              key={idx}
              className={`dataset-item ${selectedItem === idx ? 'selected' : ''}`}
              onClick={() => setSelectedItem(selectedItem === idx ? null : idx)}
            >
              <div className="dataset-item-header">
                <span className="dataset-item-number">#{idx + 1}</span>
                <span className="dataset-item-category">{item.category || 'uncategorized'}</span>
                <span className="dataset-item-question">{item.question}</span>
              </div>
              {selectedItem === idx && (
                <div className="dataset-item-details">
                  <div className="detail-section">
                    <div className="detail-label">Question:</div>
                    <div className="detail-text">{item.question}</div>
                  </div>
                  <div className="detail-section">
                    <div className="detail-label">Expected SQL:</div>
                    <pre className="detail-sql">{item.sql || item.expected_sql || 'No SQL provided'}</pre>
                  </div>
                  {item.expected_result && item.expected_result.length > 0 && (
                    <div className="detail-section">
                      <div className="detail-label">Expected Result ({item.expected_result.length} rows):</div>
                      <div className="result-table-wrapper">
                        <table className="result-table">
                          <thead>
                            <tr>
                              {Object.keys(item.expected_result[0]).map((col) => (
                                <th key={col}>{col}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {item.expected_result.slice(0, 20).map((row, i) => (
                              <tr key={i}>
                                {Object.values(row).map((val, j) => (
                                  <td key={j}>{String(val)}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {item.expected_result.length > 20 && (
                          <div className="result-truncated">
                            Showing first 20 of {item.expected_result.length} rows
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  {item.description && (
                    <div className="detail-section">
                      <div className="detail-label">Description:</div>
                      <div className="detail-text">{item.description}</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          {filteredDataset.length === 0 && (
            <div className="empty-state">
              No items found matching your filters.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default GoldenDatasetViewer

