"use client";

import { addEdge, applyEdgeChanges, applyNodeChanges, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useState } from "react";
import { CustomBidirectionalEdge } from "../components/CustomBidirectionalEdge";
import { convertDataToGraphNodesAndEdges } from "../core/data/data-converter";
import { GraphFormatService } from "../core/graph-format.service";
import { ReactFlowService } from "../core/react-flow.service";

// Register custom edge types
const edgeTypes = {
	customBidirectional: CustomBidirectionalEdge,
};

const graphFormatService = new GraphFormatService();
const reactFlowService = new ReactFlowService();

export default function App() {
	const [nodes, setNodes] = useState<any[]>([]);
	const [edges, setEdges] = useState<any[]>([]);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		const loadGraph = async () => {
			console.log('Starting graph load...');
			try {
				console.log('Converting data...');
				const {
					graphNodes,
					graphEdges,
					c1Output,
					c2Subcategories,
					c2Relationships,
					crossC1C2Relationships
				} = convertDataToGraphNodesAndEdges();

				console.log('Data converted. Nodes:', graphNodes.length, 'Edges:', graphEdges.length);
				console.log('C1:', c1Output.length, 'C2:', c2Subcategories.length);

				console.log('Starting ELK layout...');
				const layoutedData = await graphFormatService.layoutCategoriesWithNodes(
					graphNodes,
					graphEdges,
					c1Output,
					c2Subcategories,
					c2Relationships,
					crossC1C2Relationships
				);

				console.log('ELK layout complete, converting to React Flow...');
				const { nodes: initialNodes, edges: initialEdges } = reactFlowService.convertDataToReactFlowDataTypes(
					layoutedData.graphNodes,
					layoutedData.c1Nodes,
					layoutedData.c2Nodes,
					layoutedData.edges,
				);

				console.log('React Flow conversion complete. Setting state...');
				setNodes(initialNodes);
				setEdges(initialEdges);
				setIsLoading(false);
				console.log('Graph loaded successfully!');
			} catch (error) {
				console.error('Failed to load graph:', error);
				setError(error instanceof Error ? error.message : 'Unknown error');
				setIsLoading(false);
			}
		};

		loadGraph();
	}, []);

	const onNodesChange = useCallback(
		(changes: any) => setNodes((nodesSnapshot) => applyNodeChanges(changes, nodesSnapshot)),
		[],
	);
	const onEdgesChange = useCallback(
		(changes: any) => setEdges((edgesSnapshot) => applyEdgeChanges(changes, edgesSnapshot)),
		[],
	);
	const onConnect = useCallback(
		(params: any) => setEdges((edgesSnapshot) => addEdge(params, edgesSnapshot)),
		[],
	);

	if (isLoading) {
		return (
			<div style={{ width: "100vw", height: "100vh", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", background: "white" }}>
				<div style={{ marginBottom: "10px" }}>Loading graph layout...</div>
				<div style={{ fontSize: "12px", color: "#666" }}>Check browser console for details</div>
			</div>
		);
	}

	if (error) {
		return (
			<div style={{ width: "100vw", height: "100vh", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", background: "white" }}>
				<div style={{ color: "red", marginBottom: "10px" }}>Error loading graph:</div>
				<div style={{ fontSize: "12px", color: "#666" }}>{error}</div>
			</div>
		);
	}

	return (
		<div style={{ width: "100vw", height: "100vh", background: "white" }}>
			<ReactFlow
				nodes={nodes}
				edges={edges}
				edgeTypes={edgeTypes}
				onNodesChange={onNodesChange}
				onEdgesChange={onEdgesChange}
				onConnect={onConnect}
				fitView
				minZoom={0.1}
				maxZoom={2}
				style={{ background: "white" }}
			/>
		</div>
	);
}
