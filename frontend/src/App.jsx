import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import ChatInterface from './components/ChatInterface'
import ProfilingViewer from './components/ProfilingViewer'
import GoldenDatasetViewer from './components/GoldenDatasetViewer'
import MetricsDashboard from './components/MetricsDashboard'
import './App.css'

function Navigation() {
  const location = useLocation()
  
  const navItems = [
    { path: '/', label: 'Chat', icon: '💬' },
    { path: '/profiling', label: 'Profiling', icon: '📊' },
    { path: '/golden-dataset', label: 'Golden Dataset', icon: '📋' },
    { path: '/metrics', label: 'Metrics', icon: '📈' },
  ]
  
  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="nav-logo">
          <span className="logo-icon">🚀</span>
          <span className="logo-text">QueryGPT Evaluation</span>
        </div>
        <div className="nav-links">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}

function App() {
  return (
    <Router>
      <div className="app">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<ChatInterface />} />
            <Route path="/profiling" element={<ProfilingViewer />} />
            <Route path="/golden-dataset" element={<GoldenDatasetViewer />} />
            <Route path="/metrics" element={<MetricsDashboard />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App

