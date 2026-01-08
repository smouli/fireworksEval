import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './ChatInterface.css'

const CHAT_STORAGE_KEY = 'querygpt_chat_messages'
const PROVIDER_STORAGE_KEY = 'querygpt_chat_provider'

function ChatInterface() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [provider, setProvider] = useState('fireworks')
  const messagesEndRef = useRef(null)

  // Load messages from localStorage on mount
  useEffect(() => {
    const savedMessages = localStorage.getItem(CHAT_STORAGE_KEY)
    const savedProvider = localStorage.getItem(PROVIDER_STORAGE_KEY)
    
    if (savedMessages) {
      try {
        const parsedMessages = JSON.parse(savedMessages)
        // Convert timestamp strings back to Date objects
        const messagesWithDates = parsedMessages.map(msg => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
        setMessages(messagesWithDates)
      } catch (e) {
        console.error('Error loading saved messages:', e)
      }
    }
    
    if (savedProvider) {
      setProvider(savedProvider)
    }
  }, [])

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages))
    }
  }, [messages])

  // Save provider preference
  useEffect(() => {
    localStorage.setItem(PROVIDER_STORAGE_KEY, provider)
  }, [provider])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim() || loading) return

    const userMessage = {
      type: 'user',
      content: question,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    const currentQuestion = question
    setQuestion('')
    setLoading(true)

    try {
      const response = await axios.post('/api/chat', {
        question: currentQuestion,
        provider: provider,
      })

      const assistantMessage = {
        type: 'assistant',
        content: response.data.explanation || 'Query processed successfully',
        sql: response.data.sql,
        metrics: response.data.metrics,
        result: response.data.result,
        error: response.data.error,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage = {
        type: 'error',
        content: error.response?.data?.detail || error.message || 'An error occurred',
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleClearChat = () => {
    setMessages([])
    localStorage.removeItem(CHAT_STORAGE_KEY)
  }

  return (
    <div className="chat-container">
      <div className="card">
        <div className="chat-header">
          <div className="card-title">💬 Natural Language to SQL</div>
          {messages.length > 0 && (
            <button
              onClick={handleClearChat}
              className="button button-secondary clear-chat-button"
              title="Clear chat history"
            >
              Clear Chat
            </button>
          )}
        </div>
        <div className="provider-selector">
          <label>
            <input
              type="radio"
              value="fireworks"
              checked={provider === 'fireworks'}
              onChange={(e) => setProvider(e.target.value)}
            />
            Fireworks
          </label>
          <label>
            <input
              type="radio"
              value="openai"
              checked={provider === 'openai'}
              onChange={(e) => setProvider(e.target.value)}
            />
            OpenAI
          </label>
        </div>
      </div>

      <div className="card chat-messages-container">
        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <p>Ask a question in natural language to generate SQL queries.</p>
              <p className="hint">Example: "What are the names and emails of all active drivers?"</p>
            </div>
          )}
          {messages.map((msg, idx) => (
            <div key={idx} className={`message message-${msg.type}`}>
              <div className="message-header">
                <span className="message-type">
                  {msg.type === 'user' ? '👤 You' : msg.type === 'error' ? '❌ Error' : '🤖 Assistant'}
                </span>
                <span className="message-time">
                  {msg.timestamp.toLocaleTimeString()}
                </span>
              </div>
              <div className="message-content">{msg.content}</div>
              {msg.sql && (
                <div className="sql-block">
                  <div className="sql-header">Generated SQL:</div>
                  <pre className="sql-code">{msg.sql}</pre>
                </div>
              )}
              {msg.result && msg.result.length > 0 && (
                <div className="result-table-container">
                  <div className="sql-header">Query Results ({msg.result.length} rows):</div>
                  <div className="result-table-wrapper">
                    <table className="result-table">
                      <thead>
                        <tr>
                          {Object.keys(msg.result[0]).map((col) => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {msg.result.slice(0, 100).map((row, i) => (
                          <tr key={i}>
                            {Object.values(row).map((val, j) => (
                              <td key={j}>{String(val)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {msg.result.length > 100 && (
                      <div className="result-truncated">
                        Showing first 100 of {msg.result.length} rows
                      </div>
                    )}
                  </div>
                </div>
              )}
              {msg.metrics && (
                <div className="metrics-block">
                  <div className="sql-header">Performance Metrics:</div>
                  <div className="metrics-grid">
                    <div className="metric-item">
                      <span className="metric-label">Total Latency:</span>
                      <span className="metric-value">
                        {Object.values(msg.metrics).reduce(
                          (sum, m) => sum + (m.latency_ms || 0),
                          0
                        ).toFixed(0)}ms
                      </span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Total Tokens:</span>
                      <span className="metric-value">
                        {Object.values(msg.metrics).reduce(
                          (sum, m) => sum + (m.tokens || 0),
                          0
                        )}
                      </span>
                    </div>
                  </div>
                </div>
              )}
              {msg.error && (
                <div className="error-message">{msg.error}</div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message message-loading">
              <div className="loading-spinner"></div>
              <span>Processing your query...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="card chat-input-container">
        <form onSubmit={handleSubmit} className="chat-form">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question in natural language..."
            className="chat-input"
            disabled={loading}
          />
          <button
            type="submit"
            className="button button-primary chat-send-button"
            disabled={loading || !question.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}

export default ChatInterface

