import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import Webcam from "react-webcam";
import axios from "axios";
import { toast } from "react-toastify";
import { FaCamera, FaStop, FaSpinner } from "react-icons/fa";
import Sidebar from "../components/Sidebar";
import Topbar from "../components/Topbar";

export default function Attendance() {
    const navigate = useNavigate();
    const webcamRef = useRef(null);
    
    const [faculty, setFaculty] = useState(null);
    const [isActive, setIsActive] = useState(false);
    const [loading, setLoading] = useState(false);
    const [attendance, setAttendance] = useState([]);
    const [showScanning, setShowScanning] = useState(false);

    useEffect(() => {
        const storedFaculty = localStorage.getItem("faculty");
        if (!storedFaculty) {
            navigate("/");
        } else {
            setFaculty(JSON.parse(storedFaculty));
        }
    }, [navigate]);

    const stopWebcamStream = () => {
        const video = webcamRef.current?.video;
        const stream = webcamRef.current?.stream || (video && video.srcObject);

        if (stream && stream.getTracks) {
            stream.getTracks().forEach((track) => track.stop());
        }
    };

    const startAttendance = async () => {
        setLoading(true);
        setShowScanning(true);

        // Release browser webcam before backend capture begins
        stopWebcamStream();
        setIsActive(false);

        try {
            toast.info("📹 Starting face recognition... Please look at the camera.", { autoClose: 2000 });
            
            const response = await axios.get("http://127.0.0.1:5000/start-attendance", {
                timeout: 15000, // 15 second timeout
                params: {
                    faculty: faculty?.name,
                    department: faculty?.department,
                },
            });

            const message = response.data.message;
            const studentName = message.includes("for")
                ? message.split("for ")[1].trim()
                : "You";

            if (response.data.success) {
                if (response.data.alreadyMarked) {
                    toast.info(`ℹ️ ${studentName} has already been marked present today.`);
                } else {
                    toast.success(`✅ ${studentName}, attendance marked successfully!`);
                    const newEntry = {
                        id: Date.now(),
                        name: studentName,
                        time: new Date().toLocaleTimeString(),
                        status: "Present",
                    };
                    setAttendance((prevAttendance) => [newEntry, ...prevAttendance]);
                }
            } else {
                toast.error(response.data.message || "❌ Your face did not match any registered student. Please try again.");
            }
        } catch (error) {
            console.error("Attendance error:", error);
            
            if (error.code === "ECONNABORTED") {
                toast.error("❌ Request timeout. Face recognition took too long. Please try again.");
            } else if (error.message.includes("Network Error") || !error.response) {
                toast.error("❌ Cannot connect to backend server. Make sure the Python Flask server is running on port 5000.");
            } else if (error.response?.status === 400) {
                toast.error("❌ No face detected. Please position yourself properly and try again.");
            } else {
                toast.error(`❌ Error: ${error.response?.data?.error || error.message || "Unknown error"}`);
            }
        } finally {
            setLoading(false);
            setShowScanning(false);
            setIsActive(false);
        }
    };

    const stopAttendance = () => {
        setIsActive(false);
        setShowScanning(false);
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
                        {/* Header */}
                        <div className="mb-8">
                            <h2 className="text-3xl font-bold text-white mb-2">
                                Mark Attendance
                            </h2>
                            <p className="text-gray-400">
                                Date: {new Date().toLocaleDateString()}
                            </p>
                            <p className="text-gray-400">
                                Teacher: {faculty.name} ({faculty.department})
                            </p>
                        </div>

                        {/* Camera Section */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                            {/* Webcam */}
                            <div className="lg:col-span-2">
                                <div className="relative bg-black rounded-xl overflow-hidden border-2 border-blue-500/50 h-96">
                                    {isActive ? (
                                        <Webcam
                                            ref={webcamRef}
                                            className="w-full h-full object-cover"
                                            mirrored={true}
                                        />
                                    ) : (
                                        <div className="w-full h-full flex items-center justify-center">
                                            <div className="text-center">
                                                <FaCamera className="mx-auto text-gray-600 mb-4" size={48} />
                                                <p className="text-gray-400">Camera will start when you click "Start Attendance"</p>
                                            </div>
                                        </div>
                                    )}

                                    {/* Scanning Overlay */}
                                    {showScanning && (
                                        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                                            <div className="text-center">
                                                <FaSpinner className="animate-spin text-blue-400 mx-auto mb-4" size={32} />
                                                <p className="text-white font-semibold">Scanning face...</p>
                                            </div>
                                        </div>
                                    )}

                                    {/* Face Detection Frame */}
                                    {isActive && !showScanning && (
                                        <div className="absolute inset-0 pointer-events-none">
                                            <svg className="w-full h-full" viewBox="0 0 320 240">
                                                <rect x="70" y="50" width="180" height="140" fill="none" stroke="#3b82f6" strokeWidth="2" />
                                                <circle cx="160" cy="120" r="60" fill="none" stroke="#10b981" strokeWidth="1" opacity="0.5" />
                                            </svg>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Control Panel */}
                            <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6">
                                <h3 className="text-xl font-bold text-white mb-6">Instructions</h3>
                                
                                <div className="space-y-4 mb-6">
                                    <div className="flex gap-3">
                                        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center text-white text-sm">1</div>
                                        <p className="text-gray-300 text-sm">Position your face in the center</p>
                                    </div>
                                    <div className="flex gap-3">
                                        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center text-white text-sm">2</div>
                                        <p className="text-gray-300 text-sm">Ensure good lighting</p>
                                    </div>
                                    <div className="flex gap-3">
                                        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center text-white text-sm">3</div>
                                        <p className="text-gray-300 text-sm">Click "Start Attendance"</p>
                                    </div>
                                </div>

                                {/* Buttons */}
                                <div className="space-y-3">
                                    <button
                                        onClick={startAttendance}
                                        disabled={loading || isActive}
                                        className="w-full py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
                                    >
                                        {loading ? <FaSpinner className="animate-spin" /> : <FaCamera />}
                                        {loading ? "Processing..." : "Start Attendance"}
                                    </button>

                                    {isActive && (
                                        <button
                                            onClick={stopAttendance}
                                            className="w-full py-3 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg font-semibold hover:from-red-600 hover:to-red-700 transition flex items-center justify-center gap-2"
                                        >
                                            <FaStop />
                                            Stop
                                        </button>
                                    )}
                                </div>

                                {/* Status */}
                                <div className="mt-6 p-4 bg-white/5 rounded-lg border border-white/10">
                                    <p className="text-gray-400 text-xs mb-1">Status</p>
                                    <p className={`font-semibold ${isActive ? "text-green-400" : "text-gray-400"}`}>
                                        {isActive ? "● Live" : "● Inactive"}
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Attendance Records */}
                        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl overflow-hidden">
                            <div className="p-6 border-b border-white/10">
                                <h3 className="text-xl font-bold text-white">
                                    Today's Attendance ({attendance.length})
                                </h3>
                            </div>

                            {attendance.length === 0 ? (
                                <div className="p-6 text-center text-gray-400">
                                    No attendance marked yet
                                </div>
                            ) : (
                                <div className="overflow-x-auto">
                                    <table className="w-full">
                                        <thead>
                                            <tr className="bg-white/5 border-b border-white/10">
                                                <th className="px-6 py-3 text-left text-gray-300 font-semibold">Name</th>
                                                <th className="px-6 py-3 text-left text-gray-300 font-semibold">Time</th>
                                                <th className="px-6 py-3 text-left text-gray-300 font-semibold">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {attendance.map((record) => (
                                                <tr key={record.id} className="border-b border-white/5 hover:bg-white/5 transition">
                                                    <td className="px-6 py-3 text-white">{record.name}</td>
                                                    <td className="px-6 py-3 text-gray-400">{record.time}</td>
                                                    <td className="px-6 py-3">
                                                        <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-semibold">
                                                            {record.status}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
}

