import { useNavigate } from "react-router-dom";
import Header from "./Header";
import Footer from "./Footer";
import './Home.css';

const Home = () => {
    const navigate = useNavigate();

    return (
        <div className="home-page">
            <Header />

            <div className="home-title">
                <h1>K-Means Clustering vs Gaussian Mixture Models</h1>
            </div>

            <div className="next-button">
                <button
                    type="submit"
                    onClick={() => {navigate("/analysis")}}
                >
                    Next
                </button>
            </div>

            <Footer />
        </div>
    );
};

export default Home;
