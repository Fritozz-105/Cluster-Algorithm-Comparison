import { useState, useEffect } from "react";
import Papa from "papaparse";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import "./Graph.css";

interface GraphProps {
    csv: string;
    title: string;
}

const Graph: React.FC<GraphProps> = ({ csv, title }) => {
    const [chartData, setChartData] = useState<any[]>([]);

    useEffect(() => {
        const parseCSV = () => {
            Papa.parse(csv, {
                download: true,
                header: true,
                complete: (results) => {
                    const parsedData = results.data.map((row: any) => {
                        return Object.keys(row).reduce((parsedRow, key) => {
                            parsedRow[key] = isNaN(Number(row[key]))
                                ? row[key]
                                : Number(row[key]);
                            return parsedRow;
                        }, {} as any);
                    });

                    setChartData(parsedData);
                    console.log("Parsed Data:", parsedData);
                },
                skipEmptyLines: true
            });
        };

        parseCSV();
    }, [csv]);

    const getPlottableColumns = () => {
        if (chartData.length === 0) return [];

        const firstRow = chartData[0];
        return Object.keys(firstRow)
            .filter(key =>
                typeof firstRow[key] === 'number' &&
                key.toLowerCase() !== 'id'
            );
    };

    const plottableColumns = getPlottableColumns();
    console.log("Plottable Columns:", plottableColumns);

    return (
        <div className="graph-container">
            <div className="graph-wrapper">
                <h2>{title}</h2>
                {chartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={500}>
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                                dataKey={Object.keys(chartData[0])[0]}
                                allowDuplicatedCategory={false}
                            />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            {plottableColumns.map((column, index) => (
                                <Line
                                    key={column}
                                    type="monotone"
                                    dataKey={column}
                                    stroke={`hsl(${index * 60}, 70%, 50%)`}
                                    activeDot={{ r: 8 }}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                ) : (
                    <div>
                        <p>Loading data...</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Graph;
