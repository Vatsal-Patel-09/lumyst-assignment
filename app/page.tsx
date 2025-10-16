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

	useEffect(() => {
		const loadGraph = async () => {
			try {
				const {
					graphNodes,
					graphEdges,
					c1Output,
					c2Subcategories,
					c2Relationships,
					crossC1C2Relationships
				} = convertDataToGraphNodesAndEdges();

				const layoutedData = await graphFormatService.layoutCategoriesWithNodes(
					graphNodes,
					graphEdges,
					c1Output,
					c2Subcategories,
					c2Relationships,
					crossC1C2Relationships
				);

				const { nodes: initialNodes, edges: initialEdges } = reactFlowService.convertDataToReactFlowDataTypes(
					layoutedData.graphNodes,
					layoutedData.c1Nodes,
					layoutedData.c2Nodes,
					layoutedData.edges,
				);

				setNodes(initialNodes);
				setEdges(initialEdges);
				setIsLoading(false);
			} catch (error) {
				console.error('Failed to load graph:', error);
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
			<div style={{ width: "100vw", height: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "white" }}>
				<div>Loading graph layout...</div>
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
