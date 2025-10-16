import ELK, { ElkNode, ElkExtendedEdge } from 'elkjs/lib/elk.bundled.js';
import type { GraphNode, GraphEdge, C1Output, C2Subcategory, C2Relationship, CrossC1C2Relationship } from './types';

export class GraphFormatService {
	private elk: InstanceType<typeof ELK>;

	constructor() {
		this.elk = new ELK();
	}

	async layoutCategoriesWithNodes(
		graphNodes: GraphNode[],
		graphEdges: GraphEdge[],
		c1Outputs: C1Output[],
		c2Subcategories: C2Subcategory[],
		c2Relationships: C2Relationship[],
		crossC1C2Relationships: CrossC1C2Relationship[]
	) {
		console.log('[GraphFormatService] Starting layout process...');
		
		// Create a mapping from C2 names to C2 IDs for relationships
		const c2NameToIdMap = new Map();
		c2Subcategories.forEach(c2 => {
			c2NameToIdMap.set(c2.c2Name, c2.id);
		});

		console.log('[GraphFormatService] Preparing nodes...');
		// Prepare all nodes
		const allNodes = [
			...graphNodes,
			...c1Outputs.map(c1 => ({ ...c1, type: 'c1' as const })),
			...c2Subcategories.map(c2 => ({ ...c2, type: 'c2' as const }))
		];
		
		console.log(`[GraphFormatService] Total nodes: ${allNodes.length}`);

		// Prepare all edges
		const allEdges: GraphEdge[] = [
			...graphEdges,
			// Edges from C1 to their C2 subcategories
			...c2Subcategories.map(c2 => ({
				id: `c1-${c2.c1CategoryId}-to-c2-${c2.id}`,
				source: c2.c1CategoryId,
				target: c2.id,
				label: 'contains'
			})),
			// Edges from C2 to their nodes
			...c2Subcategories.flatMap(c2 =>
				c2.nodeIds.map(nodeId => ({
					id: `c2-${c2.id}-to-node-${nodeId}`,
					source: c2.id,
					target: nodeId,
					label: 'contains'
				}))
			),
			// C2 relationships
			...c2Relationships.map(rel => {
				const sourceId = c2NameToIdMap.get(rel.fromC2);
				const targetId = c2NameToIdMap.get(rel.toC2);
				if (!sourceId || !targetId) {
					return null;
				}
				return {
					id: rel.id,
					source: sourceId,
					target: targetId,
					label: rel.label
				};
			}).filter((edge): edge is GraphEdge => edge !== null),
			// Cross C1-C2 relationships
			...crossC1C2Relationships.map(rel => {
				const sourceId = c2NameToIdMap.get(rel.fromC2);
				const targetId = c2NameToIdMap.get(rel.toC2);
				if (!sourceId || !targetId) {
					return null;
				}
				return {
					id: rel.id,
					source: sourceId,
					target: targetId,
					label: rel.label
				};
			}).filter((edge): edge is GraphEdge => edge !== null)
		];

		console.log(`[GraphFormatService] Total edges: ${allEdges.length}`);
		console.log('[GraphFormatService] Calculating node connections...');

		// Helper: Calculate node degree (number of connections)
		const nodeConnections = new Map<string, { incoming: number; outgoing: number }>();
		allNodes.forEach(node => {
			nodeConnections.set(node.id, { incoming: 0, outgoing: 0 });
		});
		allEdges.forEach(edge => {
			const source = nodeConnections.get(edge.source);
			const target = nodeConnections.get(edge.target);
			if (source) source.outgoing++;
			if (target) target.incoming++;
		});

		console.log('[GraphFormatService] Converting to ELK format with layer constraints...');
		// Convert to ELK format with layer constraints
		const elkNodes: ElkNode[] = allNodes.map(node => {
			// Dynamic sizing based on node type
			let width = 150;
			let height = 50;
			let layerConstraint: string | undefined;
			let priority: number | undefined;

			const connections = nodeConnections.get(node.id);
			const isLeafNode = connections?.outgoing === 0; // No outgoing edges = leaf/end node
			const isIsolatedNode = connections?.incoming === 0 && connections?.outgoing === 0; // No connections at all

			if ('type' in node) {
				if (node.type === 'c1') {
					// C1 nodes at the top
					width = 180;
					height = 70;
					layerConstraint = 'FIRST';  // Force C1 to top layer
					priority = 100;  // Highest priority
				} else if (node.type === 'c2') {
					// C2 nodes in middle layers
					width = 160;
					height = 60;
					priority = 50;  // Medium priority
				}
			} else {
				// Regular nodes
				if (isIsolatedNode) {
					// Isolated nodes at the very bottom
					layerConstraint = 'LAST';
					priority = -100;  // Lowest priority
				} else if (isLeafNode) {
					// Leaf/end nodes near bottom
					priority = -50;  // Low priority
				} else {
					// Middle nodes
					priority = 0;  // Normal priority
				}
			}

			const elkNode: ElkNode = {
				id: node.id,
				width,
				height,
			};

			// Add layout options for layer constraints
			if (layerConstraint || priority !== undefined) {
				elkNode.layoutOptions = {};
				if (layerConstraint) {
					elkNode.layoutOptions['elk.layered.layering.layerConstraint'] = layerConstraint;
				}
				if (priority !== undefined) {
					elkNode.layoutOptions['elk.priority'] = priority.toString();
				}
			}

			return elkNode;
		});

		const elkEdges: ElkExtendedEdge[] = allEdges.map(edge => ({
			id: edge.id,
			sources: [edge.source],
			targets: [edge.target],
		}));

		// Build ELK graph with optimized layout options
		const graph: ElkNode = {
			id: 'root',
			layoutOptions: {
				'elk.algorithm': 'layered',
				'elk.direction': 'DOWN',
				
				// Spacing configuration - conservative values that preserve structure
				'elk.spacing.nodeNode': '50',           // Horizontal spacing between nodes
				'elk.layered.spacing.nodeNodeBetweenLayers': '70',  // Vertical spacing between layers
				'elk.spacing.edgeNode': '30',           // Space between edges and nodes
				'elk.spacing.edgeEdge': '15',           // Space between parallel edges
				
				// Layer assignment - ensures proper hierarchy with strict top-to-bottom ordering
				'elk.layered.layering.strategy': 'LONGEST_PATH',  // Respects layer constraints and priorities
				'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',  // Optimized node placement
				'elk.layered.considerModelOrder.strategy': 'PREFER_NODES',  // Respect node priorities
				
				// Edge routing - minimizes crossings
				'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',  // Reduces edge crossings
				'elk.edgeRouting': 'ORTHOGONAL',        // Clean 90-degree edge bends
				
				// General optimizations
				'elk.layered.thoroughness': '7',        // Balance between speed and quality (1-10)
				'elk.layered.cycleBreaking.strategy': 'GREEDY',  // Handle circular dependencies
				'elk.separateConnectedComponents': 'true',  // Keep disconnected parts separate
				
				// Padding
				'elk.padding': '[top=20,left=20,bottom=20,right=20]',
			},
			children: elkNodes,
			edges: elkEdges,
		};

		console.log(`[GraphFormatService] ELK graph prepared. Nodes: ${elkNodes.length}, Edges: ${elkEdges.length}`);
		console.log('[GraphFormatService] Running ELK layout algorithm...');

		try {
			// Run ELK layout
			const layoutedGraph = await this.elk.layout(graph);
			
			console.log('[GraphFormatService] ELK layout completed successfully!');

			// Apply positions to all nodes
			const nodePositions = new Map<string, { x: number; y: number; width: number; height: number }>();
			
			if (layoutedGraph.children) {
				for (const child of layoutedGraph.children) {
					if (child.x !== undefined && child.y !== undefined) {
						nodePositions.set(child.id, { 
							x: child.x, 
							y: child.y,
							width: child.width || 150,
							height: child.height || 50
						});
					}
				}
			}

			const positionedGraphNodes = graphNodes.map((node) => {
				const pos = nodePositions.get(node.id);
				return {
					...node,
					position: {
						x: pos?.x || 0,
						y: pos?.y || 0,
					},
				};
			});

			const positionedC1Nodes = c1Outputs.map((node) => {
				const pos = nodePositions.get(node.id);
				return {
					...node,
					position: {
						x: pos?.x || 0,
						y: pos?.y || 0,
					},
				};
			});

			const positionedC2Nodes = c2Subcategories.map((node) => {
				const pos = nodePositions.get(node.id);
				return {
					...node,
					position: {
						x: pos?.x || 0,
						y: pos?.y || 0,
					},
				};
			});

			return {
				graphNodes: positionedGraphNodes,
				c1Nodes: positionedC1Nodes,
				c2Nodes: positionedC2Nodes,
				edges: allEdges,
			};
		} catch (error) {
			console.error('ELK layout failed:', error);
			throw error;
		}
	}
}
