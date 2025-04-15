import React from 'react';

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

const ErrorState: React.FC<ErrorStateProps> = ({ message, onRetry }) => {
  return (
    <div className="error-state">
      <div className="error-icon">⚠️</div>
      <p className="error-message">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="retry-button">
          Try Again
        </button>
      )}
      
      <style jsx>{`
        .error-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 2rem;
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          min-height: 200px;
        }
        
        .error-icon {
          font-size: 2.5rem;
          margin-bottom: 1rem;
        }
        
        .error-message {
          color: #dc3545;
          text-align: center;
          margin: 0 0 1.5rem;
        }
        
        .retry-button {
          padding: 0.5rem 1.5rem;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .retry-button:hover {
          background: #0056b3;
        }
      `}</style>
    </div>
  );
};

export default ErrorState; 