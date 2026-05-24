import { Routes, Route, Navigate } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import FacultyLogin from "./pages/FacultyLogin";
import Dashboard from "./pages/Dashboard";
import Attendance from "./pages/Attendance";

// Protected Route Component
const ProtectedRoute = ({ children }) => {
    const faculty = localStorage.getItem("faculty");
    return faculty ? children : <Navigate to="/" replace />;
};

function App() {
    return (
        <div className="bg-gradient-to-br from-gray-900 via-gray-800 to-black min-h-screen text-white">
            <Routes>
                {/* Login Page */}
                <Route path="/" element={<FacultyLogin />} />

                {/* Protected Routes */}
                <Route
                    path="/dashboard"
                    element={
                        <ProtectedRoute>
                            <Dashboard />
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/attendance"
                    element={
                        <ProtectedRoute>
                            <Attendance />
                        </ProtectedRoute>
                    }
                />

                {/* Fallback */}
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>

            {/* Global Toast Notifications */}
            <ToastContainer
                position="bottom-right"
                autoClose={3000}
                hideProgressBar={false}
                newestOnTop={true}
                closeOnClick
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme="dark"
            />
        </div>
    );
}

export default App;
