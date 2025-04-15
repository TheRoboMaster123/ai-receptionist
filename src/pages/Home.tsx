import React from 'react';
import { 
  Box, 
  Button, 
  Container, 
  Grid, 
  Paper, 
  Typography,
  Theme
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import ChatIcon from '@mui/icons-material/Chat';
import SettingsIcon from '@mui/icons-material/Settings';
import type { SxProps } from '@mui/system';

const Home: React.FC = () => {
  const navigate = useNavigate();

  const paperStyle: SxProps<Theme> = {
    padding: 2,
    textAlign: 'center',
    backgroundColor: '#f5f5f5',
    cursor: 'pointer',
    '&:hover': {
      backgroundColor: '#e0e0e0',
    },
  } as const;

  return (
    <Container maxWidth="md">
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h2" component="h1" gutterBottom>
          Welcome to AI Receptionist
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          Your intelligent business assistant that handles customer inquiries 24/7
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ padding: 3 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={paperStyle} onClick={() => navigate('/business')}>
            <Box sx={{ p: 3 }}>
              <SettingsIcon sx={{ fontSize: 40, mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                Business Configuration
              </Typography>
              <Typography>
                Configure your business profile, hours, and other settings
              </Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={paperStyle} onClick={() => navigate('/chat')}>
            <Box sx={{ p: 3 }}>
              <ChatIcon sx={{ fontSize: 40, mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                Chat Interface
              </Typography>
              <Typography>
                Test and interact with your AI receptionist
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Home; 