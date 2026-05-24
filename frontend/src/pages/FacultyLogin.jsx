import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { FaUser, FaBuilding, FaSpinner } from "react-icons/fa";
import { toast } from "react-toastify";
import axios from "axios";

export default function FacultyLogin() {
    const navigate = useNavigate();
    const [form, setForm] = useState({ name: "", department: "" });
    const [loading, setLoading] = useState(false);
    const [loadingFaculties, setLoadingFaculties] = useState(true);
    const [faculties, setFaculties] = useState({});
    const [departments, setDepartments] = useState([]);
    const [facultyList, setFacultyList] = useState([]);

    // Fetch faculty list from backend on mount
    useEffect(() => {
        const fetchFaculties = async () => {
            try {
                const response = await axios.get("http://127.0.0.1:5000/api/get-faculties");
                setFaculties(response.data);
                setDepartments(Object.keys(response.data));
                setLoadingFaculties(false);
            } catch (error) {
                console.error("Error fetching faculties:", error);
                toast.error("Failed to load faculty list. Using default departments.");
                const defaultFaculties = {
                    "CSE": ["Dr. Rajesh Kumar", "Prof. Anjali Singh", "Dr. Vikas Sharma"],
                    "IT": ["Dr. Priya Gupta", "Prof. Arun Kumar", "Dr. Ravi Singh"],
                    "ECE": ["Dr. Manoj Kumar", "Prof. Sneha Verma", "Dr. Harsh Patel"],
                    "ME": ["Dr. Suresh Kumar", "Prof. Anitha Rao", "Dr. Rohan Kumar"],
                    "CE": ["Dr. Ramesh Kumar", "Prof. Deepika Singh", "Dr. Ashok Patel"],
                    "EE": ["Dr. Sundar Kumar", "Prof. Pooja Verma", "Dr. Rahul Singh"],
                    "BT": ["Dr. Vikram Kumar", "Prof. Tanvi Singh", "Dr. Sachin Patel"]
                };
                setFaculties(defaultFaculties);
                setDepartments(Object.keys(defaultFaculties));
                setLoadingFaculties(false);
            }
        };
        fetchFaculties();
    }, []);

    const handleDepartmentChange = (e) => {
        const dept = e.target.value;
        setForm({ ...form, department: dept, name: "" });
        setFacultyList(faculties[dept] || []);
    };

    const handleFacultyChange = (e) => {
        setForm({ ...form, name: e.target.value });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        setLoading(true);

        if (!form.name.trim()) {
            toast.error("Please select your name");
            setLoading(false);
            return;
        }

        if (!form.department) {
            toast.error("Please select your department");
            setLoading(false);
            return;
        }

        // Store faculty info in localStorage
        localStorage.setItem("faculty", JSON.stringify(form));
        toast.success(`Welcome, ${form.name}!`);
        
        setTimeout(() => {
            navigate("/dashboard");
        }, 500);
    };

    if (loadingFaculties) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-black">
                <FaSpinner className="animate-spin text-blue-400" size={48} />
            </div>
        );
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-black px-4">
            {/* Animated background elements */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
                <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
                <div className="absolute top-1/2 left-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
            </div>

            {/* Login Card */}
            <form
                onSubmit={handleSubmit}
                className="relative z-10 w-full max-w-md bg-white/10 backdrop-blur-md border border-white/20 p-8 rounded-2xl shadow-2xl"
            >
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 mb-2">
                        SmartAttend
                    </h1>
                    <p className="text-gray-300">Faculty Attendance System</p>
                </div>

                {/* Department Dropdown */}
                <div className="mb-6">
                    <label className="block text-gray-300 text-sm font-semibold mb-2">
                        Department
                    </label>
                    <div className="relative">
                        <FaBuilding className="absolute left-3 top-3 text-purple-400" size={18} />
                        <select
                            value={form.department}
                            onChange={handleDepartmentChange}
                            className="w-full pl-10 pr-4 py-2.5 rounded-lg bg-white/5 border border-white/20 text-white focus:outline-none focus:border-purple-400 focus:bg-white/10 transition appearance-none cursor-pointer"
                        >
                            <option value="">Select Department</option>
                            {departments.map((dept) => (
                                <option key={dept} value={dept} className="bg-gray-800">
                                    {dept}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                {/* Faculty Name Dropdown */}
                <div className="mb-6">
                    <label className="block text-gray-300 text-sm font-semibold mb-2">
                        Select Your Name
                    </label>
                    <div className="relative">
                        <FaUser className="absolute left-3 top-3 text-blue-400" size={18} />
                        <select
                            value={form.name}
                            onChange={handleFacultyChange}
                            disabled={!form.department}
                            className="w-full pl-10 pr-4 py-2.5 rounded-lg bg-white/5 border border-white/20 text-white focus:outline-none focus:border-blue-400 focus:bg-white/10 transition appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <option value="">
                                {form.department ? "Select your name" : "Select department first"}
                            </option>
                            {facultyList.map((faculty) => (
                                <option key={faculty} value={faculty} className="bg-gray-800">
                                    {faculty}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                {/* Login Button */}
                <button
                    type="submit"
                    disabled={loading || !form.name || !form.department}
                    className="w-full py-2.5 mt-8 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold hover:from-blue-600 hover:to-purple-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? "Logging in..." : "Login"}
                </button>

                {/* Footer */}
                <p className="text-center text-gray-400 text-sm mt-6">
                    Using AI-Powered Face Recognition Technology
                </p>
            </form>
        </div>
    );
}
