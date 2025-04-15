import React, { useState, useEffect } from 'react';
import { Line, Pie, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import { format } from 'date-fns';
import LoadingState from './LoadingState';
import ErrorState from './ErrorState';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface AnalyticsDashboardProps {
  businessId: string;
}

interface TimeRange {
  startDate: Date;
  endDate: Date;
}

const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({ businessId }) => {
  const [timeRange, setTimeRange] = useState<TimeRange>({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
    endDate: new Date()
  });
  const [interval, setInterval] = useState<'hour' | 'day' | 'week'>('day');
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const [sentimentData, setSentimentData] = useState<any>(null);
  const [topicData, setTopicData] = useState<any>(null);
  const [volumeData, setVolumeData] = useState<any>(null);
  const [responseTimeData, setResponseTimeData] = useState<any>(null);
  
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch all visualization data
      const [sentiment, topics, volume, responseTimes] = await Promise.all([
        fetch(`/api/visualizations/business/${businessId}/sentiment?start_date=${timeRange.startDate.toISOString()}&end_date=${timeRange.endDate.toISOString()}&interval=${interval}`),
        fetch(`/api/visualizations/business/${businessId}/topics?start_date=${timeRange.startDate.toISOString()}&end_date=${timeRange.endDate.toISOString()}`),
        fetch(`/api/visualizations/business/${businessId}/volume?start_date=${timeRange.startDate.toISOString()}&end_date=${timeRange.endDate.toISOString()}&interval=${interval}`),
        fetch(`/api/visualizations/business/${businessId}/response-times?start_date=${timeRange.startDate.toISOString()}&end_date=${timeRange.endDate.toISOString()}`)
      ]).then(responses => Promise.all(responses.map(async r => {
        if (!r.ok) {
          throw new Error(`API error: ${r.statusText}`);
        }
        return r.json();
      })));
      
      setSentimentData(sentiment);
      setTopicData(topics);
      setVolumeData(volume);
      setResponseTimeData(responseTimes);
    } catch (error) {
      console.error('Error fetching analytics data:', error);
      setError('Failed to load analytics data. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchData();
  }, [businessId, timeRange, interval]);
  
  if (loading) {
    return <LoadingState />;
  }
  
  if (error) {
    return <ErrorState message={error} onRetry={fetchData} />;
  }
  
  return (
    <div className="analytics-dashboard">
      <div className="analytics-controls">
        <div className="time-range-selector">
          <label>Time Range:</label>
          <select
            value={interval}
            onChange={(e) => setInterval(e.target.value as 'hour' | 'day' | 'week')}
          >
            <option value="hour">Hourly</option>
            <option value="day">Daily</option>
            <option value="week">Weekly</option>
          </select>
        </div>
      </div>
      
      <div className="analytics-grid">
        {/* Sentiment Timeline */}
        <div className="analytics-card">
          <h3>Sentiment Over Time</h3>
          {sentimentData && (
            <Line
              data={sentimentData.data}
              options={{
                responsive: true,
                plugins: {
                  title: {
                    display: true,
                    text: 'Conversation Sentiment Trends'
                  },
                  tooltip: {
                    mode: 'index',
                    intersect: false,
                  }
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                      display: true,
                      text: 'Sentiment Score'
                    }
                  }
                }
              }}
            />
          )}
        </div>
        
        {/* Topic Distribution */}
        <div className="analytics-card">
          <h3>Topic Distribution</h3>
          {topicData && (
            <Pie
              data={topicData.data}
              options={{
                responsive: true,
                plugins: {
                  title: {
                    display: true,
                    text: 'Conversation Topics'
                  },
                  legend: {
                    position: 'right'
                  }
                }
              }}
            />
          )}
        </div>
        
        {/* Conversation Volume */}
        <div className="analytics-card">
          <h3>Conversation Volume</h3>
          {volumeData && (
            <Bar
              data={volumeData.data}
              options={{
                responsive: true,
                plugins: {
                  title: {
                    display: true,
                    text: 'Conversation Volume by Status'
                  },
                  tooltip: {
                    mode: 'index',
                    intersect: false,
                  }
                },
                scales: {
                  x: {
                    stacked: true,
                  },
                  y: {
                    stacked: true,
                    title: {
                      display: true,
                      text: 'Number of Conversations'
                    }
                  }
                }
              }}
            />
          )}
        </div>
        
        {/* Response Times */}
        <div className="analytics-card">
          <h3>Response Time Distribution</h3>
          {responseTimeData && (
            <Bar
              data={responseTimeData.data}
              options={{
                responsive: true,
                plugins: {
                  title: {
                    display: true,
                    text: 'Response Time Distribution'
                  }
                },
                scales: {
                  y: {
                    title: {
                      display: true,
                      text: 'Frequency'
                    }
                  },
                  x: {
                    title: {
                      display: true,
                      text: 'Response Time'
                    }
                  }
                }
              }}
            />
          )}
        </div>
      </div>
      
      <style jsx>{`
        .analytics-dashboard {
          padding: 2rem;
          background: #f8f9fa;
        }
        
        .analytics-controls {
          margin-bottom: 2rem;
          padding: 1rem;
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .time-range-selector {
          display: flex;
          align-items: center;
          gap: 1rem;
        }
        
        .time-range-selector select {
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
        }
        
        .analytics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
          gap: 2rem;
        }
        
        .analytics-card {
          background: white;
          padding: 1.5rem;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .analytics-card h3 {
          margin-top: 0;
          margin-bottom: 1.5rem;
          color: #333;
          font-size: 1.2rem;
        }
      `}</style>
    </div>
  );
};

export default AnalyticsDashboard; 