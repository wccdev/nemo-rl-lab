import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { AuthProvider, useAuth } from './context/AuthContext'
import { ThemeProvider } from './context/ThemeContext'
import { ComparePage } from './pages/Compare'
import { DashboardPage } from './pages/Dashboard'
import { ExperimentsPage } from './pages/Experiments'
import { JobDetailPage } from './pages/JobDetail'
import { JobsPage } from './pages/Jobs'
import { LoginPage } from './pages/Login'
import { RunsPage } from './pages/Runs'

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { user, loading, noAuth } = useAuth()
  if (loading) return <div className="flex min-h-screen items-center justify-center text-muted">加载中…</div>
  if (!noAuth && !user) return <Navigate to="/login" replace />
  return children
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route
              element={
                <PrivateRoute>
                  <Layout />
                </PrivateRoute>
              }
            >
              <Route index element={<DashboardPage />} />
              <Route path="jobs" element={<JobsPage />} />
              <Route path="jobs/:id" element={<JobDetailPage />} />
              <Route path="compare" element={<ComparePage />} />
              <Route path="runs" element={<RunsPage />} />
              <Route path="experiments" element={<ExperimentsPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  )
}
