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
    const [yAxisColumn, setYAxisColumn] = useState<string>('');
    const [clusterColumn, setClusterColumn] = useState<string>('');

    useEffect(() => {
        const parseCSV = () => {
            Papa.parse(csv, {
                header: true,
                dynamicTyping: true,
                complete: (results) => {
                    if (results.data.length === 0) {
                        console.error("No data parsed from CSV");
                        return;
                    }

                    // Determine columns
                    const columns = Object.keys(results.data[0]);
                    const numericColumns = columns.filter(
                        key => typeof results.data[0][key] === 'number'
                    );

                    // Find t-SNE columns and cluster column
                    const tsneColumns = columns.filter(
                        col => col.includes('t-SNE')
                    );
                    const clusterCol = columns.find(
                        col => col.toLowerCase().includes('cluster')
                    );

                    if (tsneColumns.length >= 2 && clusterCol) {
                        setXAxisColumn(tsneColumns[0]);
                        setYAxisColumn(tsneColumns[1]);
                        setClusterColumn(clusterCol);

                        // Color coding based on cluster
                        const colorMap: { [key: number]: string } = {
                            0: 'hsl(180, 70%, 50%)',   // Blue for cluster 0
                            1: 'hsl(340, 70%, 50%)'    // Red for cluster 1
                        };

                        const coloredData = results.data.map(row => ({
                            ...row,
                            color: colorMap[row[clusterCol]] || 'gray'
                        }));

                        setChartData(coloredData);
                    } else {
                        console.error("Could not find appropriate columns", { columns, tsneColumns, clusterCol });
                    }
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
                                type="number"
                                dataKey={xAxisColumn}
                                name={xAxisColumn}
                                label={{
                                    value: xAxisTitle || xAxisColumn,
                                    position: 'insideBottomRight',
                                    offset: -5
                                }}
                            />
                            <YAxis
                                type="number"
                                dataKey={yAxisColumn}
                                name={yAxisColumn}
                                label={{
                                    value: yAxisTitle || yAxisColumn,
                                    angle: -90,
                                    position: 'insideLeft'
                                }}
                            />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                            <Legend />
                            <Scatter
                                name="Data Points"
                                data={chartData}
                                fill={chartData[0]?.color || 'hsl(180, 70%, 50%)'}
                                strokeOpacity={0.7}
                                fillOpacity={0.6}
                                shape="circle"
                                strokeWidth={1}
                                r={2}
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
