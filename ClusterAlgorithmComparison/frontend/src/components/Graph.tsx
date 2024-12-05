import React, { useState, useEffect } from "react";
import Papa from "papaparse";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import "./Graph.css";

interface GraphProps {
    csv: string;
    title: string;
    xAxisTitle?: string;
    yAxisTitle?: string;
}

const Graph: React.FC<GraphProps> = ({ csv, title, xAxisTitle, yAxisTitle }) => {
    const [chartData, setChartData] = useState<any[]>([]);
    const [xAxisColumn, setXAxisColumn] = useState<string>('');

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

                    const numericColumns = Object.keys(parsedData[0])
                        .filter(key => typeof parsedData[0][key] === 'number' && key.toLowerCase() !== 'id');

                    if (numericColumns.length > 0) {
                        const firstNumericColumn = numericColumns[0];
                        setXAxisColumn(firstNumericColumn);

                        const sortedData = [...parsedData].sort((a, b) =>
                            (a[firstNumericColumn] || 0) - (b[firstNumericColumn] || 0)
                        );

                        setChartData(sortedData);
                    } else {
                        setChartData(parsedData);
                    }

                    console.log("Parsed Data:", parsedData);
                },
                skipEmptyLines: true
            });
        };

        parseCSV();
    }, [csv]);

    return (
        <div className="graph-container">
            <div className="graph-wrapper">
                <h2>{title}</h2>
                {chartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={500}>
                        <ScatterChart>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                                dataKey={xAxisColumn}
                                type="number"
                                name={xAxisColumn}
                                label={{
                                    value: xAxisTitle || xAxisColumn,
                                    position: 'insideBottomRight',
                                    offset: -5
                                }}
                            />
                            <YAxis
                                dataKey={Object.keys(chartData[0]).find(key =>
                                    typeof chartData[0][key] === 'number' && key !== xAxisColumn
                                )}
                                type="number"
                                name={yAxisTitle}
                                label={{
                                    value: yAxisTitle || 'Y-Axis',
                                    angle: -90,
                                    position: 'insideLeft'
                                }}
                            />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                            <Legend />
                            <Scatter
                                name="Data Points"
                                data={chartData}
                                fill="hsl(180, 70%, 50%)"
                            />
                        </ScatterChart>
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
