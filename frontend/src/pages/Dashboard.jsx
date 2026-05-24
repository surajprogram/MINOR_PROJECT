import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import Sidebar from "../components/Sidebar";
import Topbar from "../components/Topbar";
import { FaClipboardList, FaCamera, FaCheckCircle, FaDownload } from "react-icons/fa";
import { toast } from "react-toastify";

export default function Dashboard() {
    const navigate = useNavigate();
    const [faculty, setFaculty] = useState(null);
    const [downloadLoading, setDownloadLoading] = useState(false);
    const [stats, setStats] = useState({
        totalMarked: 0,
        todayMarked: 0,
    });

    useEffect(() => {
        const storedFaculty = localStorage.getItem("faculty");
        if (!storedFaculty) {
            navigate("/");
        } else {
            setFaculty(JSON.parse(storedFaculty));
        }
    }, [navigate]);

    const handleDownloadAttendance = async () => {
        setDownloadLoading(true);
        try {
            const response = await axios.get(
                "http://127.0.0.1:5000/api/download-attendance",
                {
                    responseType: "blob",
                }
            );
            
            // Create a blob URL and trigger download
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement("a");
            link.href = url;
            link.setAttribute(
                "download",
                `attendance_${new Date().toISOString().split("T")[0]}.xlsx`
            );
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);
            window.URL.revokeObjectURL(url);
            
            toast.success("✅ Attendance sheet downloaded successfully!");
        } catch (error) {
            console.error("Download error:", error);
            toast.error("❌ No attendance records found or download failed");
        } finally {
            setDownloadLoading(false);
        }
    };

    if (!faculty) return null;

    return (
        <div className="flex h-screen bg-gray-900">
            <Sidebar />
            <div className="flex-1 flex flex-col overflow-hidden">
                <Topbar faculty={faculty} />
                
                {/* Main Content */}
                <main className="flex-1 overflow-auto bg-gradient-to-br from-gray-900 via-gray-800 to-black">
                    <div className="p-8">
                        {/* Welcome Section */}
                        <div className="mb-8">
                            <h2 className="text-3xl font-bold text-white mb-2">
                                Welcome back, {faculty.name}! 👋
                            </h2>
                            <p className="text-gray-400">
                                Department: <span className="text-blue-400 font-semibold">{faculty.department}</span>
                            </p>
                        </div>

                        {/* Stats Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                            {/* Mark Attendance Card */}
                            <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/20 transition cursor-pointer" onClick={() => navigate("/attendance")}>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-gray-400 text-sm mb-2">Mark Attendance</p>
                                        <h3 className="text-2xl font-bold text-white">Face Recognition</h3>
                                    </div>
                                    <FaCamera className="text-blue-400" size={32} />
                                </div>
                            </div>

                            {/* Today Marked Card */}
                            <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-gray-400 text-sm mb-2">Today's Attendance</p>
                                        <h3 className="text-2xl font-bold text-white">{stats.todayMarked}</h3>
                                    </div>
                                    <FaCheckCircle className="text-green-400" size={32} />
                                </div>
                            </div>

                            {/* Total Marked Card */}
                            <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-gray-400 text-sm mb-2">Total Marked</p>
                                        <h3 className="text-2xl font-bold text-white">{stats.totalMarked}</h3>
                                    </div>
                                    <FaClipboardList className="text-purple-400" size={32} />
                                </div>
                            </div>
                        </div>

                        {/* Action Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                            {/* Quick Start */}
                            <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6">
                                <h3 className="text-xl font-bold text-white mb-4">📸 Quick Start</h3>
                                <p className="text-gray-400 mb-4">
                                    Click below to start marking attendance using face recognition.
                                </p>
                                <button
                                    onClick={() => navigate("/attendance")}
                                    className="w-full px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition flex items-center justify-center gap-2"
                                >
                                    <FaCamera size={16} />
                                    Go to Attendance
                                </button>
                            </div>

                            {/* Download Attendance */}
                            <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6">
                                <h3 className="text-xl font-bold text-white mb-4">📊 Download Sheet</h3>
                                <p className="text-gray-400 mb-4">
                                    Download attendance records as an Excel file.
                                </p>
                                <button
                                    onClick={handleDownloadAttendance}
                                    disabled={downloadLoading}
                                    className="w-full px-6 py-2 bg-gradient-to-r from-green-500 to-teal-600 text-white rounded-lg hover:from-green-600 hover:to-teal-700 transition flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    <FaDownload size={16} />
                                    {downloadLoading ? "Downloading..." : "Download Excel"}
                                </button>
                            </div>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
}
