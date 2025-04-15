import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  CircularProgress,
  Alert,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from 'recharts';

interface MetricsData {
  timestamp: string;
  conversation_metrics: {
    conversation_counts: {
      active: number;
      inactive: number;
      archived: number;
    };
    avg_messages_per_conversation: number;
    summary_stats: {
      total_summaries: number;
      avg_summary_length: number;
    };
  };
  cleanup_performance: {
    period_hours: number;
    conversations_processed: {
      marked_inactive: number;
      archived: number;
    };
    messages_summarized: number;
  };
  memory_efficiency: {
    total_messages: number;
    summarized_messages: number;
    compression_ratio: number;
    memory_saved_percentage: number;
  };
}

interface ResourceMetrics {
  current: {
    gpu_usage: number;
    cpu_usage: number;
    memory_usage: number;
    active_requests: number;
    max_concurrent_requests: number;
  };
  request_stats: {
    average_duration: number;
    average_gpu_memory: number;
    requests_processed: number;
    requests_per_minute: number;
  };
  thresholds: {
    gpu_memory: number;
    cpu: number;
  };
  last_update: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

export const MetricsDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<number>(24);
  const [resourceMetrics, setResourceMetrics] = useState<ResourceMetrics | null>(null);
  const [resourceError, setResourceError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/metrics/cleanup?period_hours=${timeRange}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('admin_token')}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch metrics');
      }
      
      const data = await response.json();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchResourceMetrics = async () => {
    try {
      const response = await fetch('/api/metrics/resources?minutes=5', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('admin_token')}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch resource metrics');
      }
      
      const data = await response.json();
      setResourceMetrics(data);
      setResourceError(null);
    } catch (err) {
      setResourceError(err.message);
    }
  };

  useEffect(() => {
    fetchMetrics();
    fetchResourceMetrics();
    // Refresh metrics every 5 minutes, resources every 10 seconds
    const metricsInterval = setInterval(fetchMetrics, 300000);
    const resourcesInterval = setInterval(fetchResourceMetrics, 10000);
    return () => {
      clearInterval(metricsInterval);
      clearInterval(resourcesInterval);
    };
  }, [timeRange]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Error loading metrics: {error}
      </Alert>
    );
  }

  if (!metrics) {
    return null;
  }

  const conversationData = [
    { name: 'Active', value: metrics.conversation_metrics.conversation_counts.active },
    { name: 'Inactive', value: metrics.conversation_metrics.conversation_counts.inactive },
    { name: 'Archived', value: metrics.conversation_metrics.conversation_counts.archived },
  ];

  const ResourceUsageCard = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Resource Usage
        </Typography>
        {resourceError ? (
          <Alert severity="error" sx={{ mb: 2 }}>
            Error loading resource metrics: {resourceError}
          </Alert>
        ) : resourceMetrics ? (
          <Box>
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                GPU Memory Usage
              </Typography>
              <LinearProgress
                variant="determinate"
                value={resourceMetrics.current.gpu_usage * 100}
                color={resourceMetrics.current.gpu_usage > resourceMetrics.thresholds.gpu_memory ? "error" : "primary"}
                sx={{ height: 10, borderRadius: 5 }}
              />
              <Typography variant="caption">
                {(resourceMetrics.current.gpu_usage * 100).toFixed(1)}%
              </Typography>
            </Box>

            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                CPU Usage
              </Typography>
              <LinearProgress
                variant="determinate"
                value={resourceMetrics.current.cpu_usage * 100}
                color={resourceMetrics.current.cpu_usage > resourceMetrics.thresholds.cpu ? "error" : "primary"}
                sx={{ height: 10, borderRadius: 5 }}
              />
              <Typography variant="caption">
                {(resourceMetrics.current.cpu_usage * 100).toFixed(1)}%
              </Typography>
            </Box>

            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Memory Usage
              </Typography>
              <LinearProgress
                variant="determinate"
                value={resourceMetrics.current.memory_usage * 100}
                sx={{ height: 10, borderRadius: 5 }}
              />
              <Typography variant="caption">
                {(resourceMetrics.current.memory_usage * 100).toFixed(1)}%
              </Typography>
            </Box>

            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2">
                Active Requests: {resourceMetrics.current.active_requests} / {resourceMetrics.current.max_concurrent_requests}
              </Typography>
            </Box>

            <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>
              Request Statistics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Avg Duration: {resourceMetrics.request_stats.average_duration.toFixed(2)}s
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Requests/min: {resourceMetrics.request_stats.requests_per_minute.toFixed(1)}
                </Typography>
              </Grid>
            </Grid>

            <Typography variant="caption" display="block" sx={{ mt: 2, textAlign: 'right' }}>
              Last updated: {new Date(resourceMetrics.last_update).toLocaleTimeString()}
            </Typography>
          </Box>
        ) : (
          <CircularProgress />
        )}
      </CardContent>
    </Card>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4">System Metrics Dashboard</Typography>
        <Box>
          <FormControl sx={{ minWidth: 120, mr: 2 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => setTimeRange(Number(e.target.value))}
            >
              <MenuItem value={24}>Last 24 Hours</MenuItem>
              <MenuItem value={72}>Last 3 Days</MenuItem>
              <MenuItem value={168}>Last Week</MenuItem>
            </Select>
          </FormControl>
          <Button variant="contained" onClick={fetchMetrics}>
            Refresh
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Resource Usage */}
        <Grid item xs={12} md={4}>
          <ResourceUsageCard />
        </Grid>

        {/* Conversation Status */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Conversation Status
              </Typography>
              <Box height={300}>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie
                      data={conversationData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label
                    >
                      {conversationData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Memory Efficiency */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Memory Efficiency
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body1">
                  Compression Ratio: {metrics.memory_efficiency.compression_ratio}
                </Typography>
                <Typography variant="body1">
                  Memory Saved: {metrics.memory_efficiency.memory_saved_percentage}%
                </Typography>
              </Box>
              <Box height={200}>
                <ResponsiveContainer>
                  <LineChart data={[
                    { name: 'Total', value: metrics.memory_efficiency.total_messages },
                    { name: 'Summarized', value: metrics.memory_efficiency.summarized_messages }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="value" stroke="#8884d8" />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Cleanup Performance */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Cleanup Performance
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Typography variant="body1">
                    Messages Summarized: {metrics.cleanup_performance.messages_summarized}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body1">
                    Conversations Archived: {metrics.cleanup_performance.conversations_processed.archived}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body1">
                    Marked Inactive: {metrics.cleanup_performance.conversations_processed.marked_inactive}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Request Performance */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Request Performance
              </Typography>
              <Box height={300}>
                <ResponsiveContainer>
                  <AreaChart
                    data={[
                      {
                        name: 'Current',
                        requests: resourceMetrics?.current.active_requests || 0,
                        duration: resourceMetrics?.request_stats.average_duration || 0,
                      }
                    ]}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Legend />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="requests"
                      stroke="#8884d8"
                      fill="#8884d8"
                      name="Active Requests"
                    />
                    <Area
                      yAxisId="right"
                      type="monotone"
                      dataKey="duration"
                      stroke="#82ca9d"
                      fill="#82ca9d"
                      name="Avg Duration (s)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}; 