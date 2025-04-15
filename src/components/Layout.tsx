import React from 'react';
import { AppBar, Box, Container, Toolbar, Typography } from '@mui/material';
import { Link } from 'react-router-dom';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static">
        <Toolbar>
          <Typography 
            variant="h6" 
            component={Link} 
            to="/" 
            sx={{ 
              textDecoration: 'none', 
              color: 'inherit',
              flexGrow: 1 
            }}
          >
            AI Receptionist
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Typography 
              component={Link} 
              to="/chat" 
              sx={{ 
                textDecoration: 'none', 
                color: 'inherit' 
              }}
            >
              Chat
            </Typography>
            <Typography 
              component={Link} 
              to="/setup" 
              sx={{ 
                textDecoration: 'none', 
                color: 'inherit' 
              }}
            >
              Business Setup
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>
      <Container 
        component="main" 
        maxWidth="lg" 
        sx={{ 
          flexGrow: 1, 
          py: 4 
        }}
      >
        {children}
      </Container>
      <Box 
        component="footer" 
        sx={{ 
          py: 3, 
          px: 2, 
          mt: 'auto', 
          backgroundColor: (theme) => theme.palette.grey[200] 
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="body2" color="text.secondary" align="center">
            © {new Date().getFullYear()} AI Receptionist. All rights reserved.
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};

export default Layout; 