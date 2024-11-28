import { useNavigate } from "react-router-dom";
import "./Header.css"
import logo from "../assets/logo.jpg";

const Header = () => {
    const navigate = useNavigate();

    return (
        <header className="header">
            <div className="header-content">
                <img
                    src={logo}
                    alt="logo"
                    onClick={() => {navigate("/")}}
                    className="logo-image"
                />
            </div>
        </header>
    );
};

export default Header;
