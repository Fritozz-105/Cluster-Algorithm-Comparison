import { useLocation } from "react-router-dom";
import "./Analysis.css";
import Header from "./Header";
import Footer from "./Footer";
import Graph from "./Graph";
import Data1 from "../assets/sampledata1.csv?url";
import Data2 from "../assets/sampledata2.csv?url";

const Analysis = () => {
    const location = useLocation();
    const { input1, input2 } = location.state || {};

    return (
        <div className="analysis-content">
            <Header />

            <div className="analysis-title">
                <h1>Analysis</h1>
            </div>
            <div className="analysis-form">
                <h5>Input 1</h5>
                <p>{input1}</p>
                <h5>Input 2</h5>
                <p>{input2}</p>
            </div>

            <div className="graph-section">
                <Graph csv={Data1} title="K-Means"/>
                <Graph csv={Data2} title="Gaussian"/>
            </div>

            <Footer />
        </div>
    );
};

export default Analysis;
