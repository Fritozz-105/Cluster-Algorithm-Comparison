import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "./Header";
import Footer from "./Footer";
import './Home.css';

const Home = () => {
    const navigate = useNavigate();
    const [formInput, setFormInput] = useState({
        input1: "",
        input2: "",
    });
    const handleSubmit = (e) => {
        e.preventDefault();

        navigate("/analysis", {
            state: {
            ... formInput,
            input1: formInput.input1,
            input2: formInput.input2
            }
        });
    };

    return (
        <div className="home-page">

            <Header />

            <div className="home-title">
                <h1>K-Means Clustering vs Gaussian Mixture Models</h1>
            </div>
            <div className="home-form">
                <h5>Input 1</h5>
                <input
                    type="text"
                    value={formInput.input1}
                    onChange={(e) => setFormInput({ ...formInput, input1: e.target.value })}
                />
                <h5>Input 2</h5>
                <input
                    type="text"
                    value={formInput.input2}
                    onChange={(e) => setFormInput({ ...formInput, input2: e.target.value })}
                />
            </div>
            <div className="next-button">
                <button onClick={handleSubmit}>
                    Next
                </button>
            </div>

            <Footer />

        </div>
    );
};

export default Home;
